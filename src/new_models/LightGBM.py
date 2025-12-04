import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils import graficar_x_pred_mensual,escalar_asinh_vector,makewindows,putTest_cuenta,predictionTest

# --- helpers para la misma normalización que usaste al entrenar ---
def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s
class LightGBM_TweedieSeriesWrapper:
    """
    Versión 'como TCN/MLP' pero usando LightGBM Tweedie.
    Recibe la SERIE (diaria/mensual, una cuenta), hace asinh+MinMax,
    hace ventanas, aplana y entrena LightGBM.
    Luego arma el tramo de test con fechas reales y devuelve también mensual.
    """
    def __init__(self, tweedie_variance_power=1.5, **params):
        base = dict(
            objective='tweedie',
            tweedie_variance_power=tweedie_variance_power,
            learning_rate=0.03,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_data_in_leaf=12,
            min_gain_to_split=0.0,
            n_estimators=400,
            random_state=42,
        )
        base.update(params)
        self.lgbm = lgb.LGBMRegressor(**base)

        # scaler para las features (las ventanas ya asinh+minmax)
        self.scaler_X = StandardScaler()

        # estos los llenamos en train
        self.scaler = None      # MinMaxScaler del asinh
        self.s = None           # escala robusta del asinh
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None   # aquí guardamos una sola corrida (lista de 1) por compatibilidad
        self.colname = None

    # =========================================================
    # 1) API para tu for: recibe la serie y regresa mensual
    # =========================================================
    def train_from_series(self,
                          x,
                          train_start=dt.datetime(2013, 1, 1),
                          train_end=dt.datetime(2024, 1, 1),
                          colname=None):
        """
        Igual que con TCNWrapper/MLPWrapper: tú le pasas la serie (ya aumentada)
        y él hace todo por dentro.
        """
        res = self.train(
            x,
            colname=colname,
            test_start=dt.datetime(2024, 2, 1),
            test_end=dt.datetime(2025, 6, 1),
        )

        # convertir su salida diaria a mensual para dejar todo homogéneo
        pred_daily = pd.Series(
            np.asarray(res["y_pred"]).ravel(),
            index=self.test_df.index
        )
        pred_month = pred_daily.resample("MS").mean()
        real_month = self.test_df["Adj Close"].resample("MS").mean()

        common_idx = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[common_idx]
        real_month = real_month.loc[common_idx]

        mse = float(((real_month.values - pred_month.values) ** 2).mean())
        rmse = float(np.sqrt(mse))

        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": mse,
            "RMSE": rmse,
        }

    # =========================================================
    # 2) train "real" (como tu TCN pero con LightGBM)
    # =========================================================
    def train(self,
              x,
              colname=None,
              test_start=dt.datetime(2024, 2, 1),
              test_end=dt.datetime(2025, 6, 1)):
        """
        x: Serie/DataFrame de UNA cuenta (puede venir diaria por tu aumento)
        """
        # --- nombre ---
        if colname is None:
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # --- asegurar (N,1) ---
        if isinstance(x, pd.DataFrame):
            serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
        else:
            serie = pd.Series(x).astype(float).values.reshape(-1, 1)

        # --- asinh + MinMax split (lo mismo que usas para TCN) ---
        train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
            serie,
            train_ratio=0.8
        )

        # --- ventanas ---
        # X_train: (N, lookback, 1)
        X_train, y_train, X_test, y_test, prediction_days = makewindows(
            train_data, test_data
        )
        self.prediction_days = prediction_days

        # LightGBM quiere 2D → aplanamos
        N_tr, L, F = X_train.shape
        X_train_flat = X_train.reshape(N_tr, L * F)

        # target no puede tener negativos para tweedie, pero aquí y_train
        # viene en [0,1] porque viene del MinMax después del asinh → OK
        y = np.asarray(y_train, float).ravel()
        if np.any(y < 0):
            raise ValueError("LightGBM Tweedie: target < 0 después de escalar. Revisa el pipeline.")
        if np.all(y == 0):
            raise ValueError("LightGBM Tweedie: todo 0 → no entreno.")

        # escalar features para LightGBM (esto es tu estilo de wrapper original)
        Xs = self.scaler_X.fit_transform(X_train_flat)

        # entrenar
        self.lgbm.fit(Xs, y)

        # --- ahora preparamos el tramo de test con fechas reales ---
        model_inputs, test_df, t_start, t_end = putTest_cuenta(
            x,
            prediction_days=prediction_days,
            scaler=scaler,
            s=s,
            test_start=test_start,
            test_end=test_end,
        )

        # construir ventanas de test igual que las de train
        x_test_seq = []
        for i in range(prediction_days, len(model_inputs)):
            x_test_seq.append(model_inputs[i - prediction_days:i, 0])
        x_test_seq = np.array(x_test_seq)   # (N_test, lookback)

        # escalar con el scaler_X de LightGBM
        x_test_seq_s = self.scaler_X.transform(x_test_seq)

        # predecir en espacio asinh+MinMax
        preds_scaled = self.lgbm.predict(x_test_seq_s).reshape(-1, 1)

        # desescalar (MinMax^{-1} + des-asinh)
        preds = inverse_scale(preds_scaled, s, scaler)

        # métricas contra el real diario
        y_true = test_df[["Adj Close"]].values
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))

        # guardar todo
        self.scaler = scaler
        self.s = s
        self.test_df = test_df
        self.test_start = t_start
        self.test_end = t_end
        # lo guardo como lista para parecerme al TCN/MLP
        self.all_preds = [preds.squeeze()]

        return {
            "y_true": y_true.squeeze(),
            "y_pred": preds.squeeze(),
            "MSE": float(mse),
            "RMSE": rmse,
        }

    # =========================================================
    def evaluate(self, model=None, X_test=None, y_test=None):
        """
        Como en TCNWrapper/MLPWrapper: devolvemos mensual usando lo que
        se guardó en train().
        """
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) o .train_from_series(...)")

        # tenemos una sola corrida, pero lo medio uniformamos
        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index)

        pred_month = pred_daily.resample("MS").mean()
        real_month = self.test_df["Adj Close"].resample("MS").mean()

        common_idx = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[common_idx]
        real_month = real_month.loc[common_idx]

        mse = float(((real_month.values - pred_month.values) ** 2).mean())
        rmse = float(np.sqrt(mse))

        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": mse,
            "RMSE": rmse,
        }

    # =========================================================
    def predecir_futuro(self, *args, **kwargs):
        """
        Igual que en los otros wrappers de serie: devolvemos lo mensual
        que ya calculamos del tramo de test.
        """
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) / .train_from_series(...)")
        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index)
        pred_month = pred_daily.resample("MS").mean()
        return pred_month.values

    # =========================================================
    def plot(self, x, end_limit="2025-06-01"):
        """
        Solo si ya tienes tu función graficar_x_pred_mensual(...)
        """
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Entrena primero con .train(...)")
        _ = graficar_x_pred_mensual(
            x=x,
            all_preds=self.all_preds,
            test_df=self.test_df,
            test_start=self.test_start,
            end_limit=end_limit,
            titulo=f"Cuenta {self.colname}: Real mensual vs Pred mensual (LightGBM Tweedie)"
        )
