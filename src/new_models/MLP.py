import datetime as dt
import numpy as np
import pandas as pd
from src.utils import graficar_x_pred_mensual,escalar_asinh_vector,makewindows,putTest_cuenta,predictionTest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# --- helpers para la misma normalización que usaste al entrenar ---
def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s

class MLPSeriesWrapper:
    """
    Mismo patrón que tu TCNWrapper/LSTMSeries:
    - recibe la serie (diaria o mensual aumentada)
    - hace asinh + MinMax + ventanas
    - MLP (sklearn) trabaja en 2D, así que aplanamos las ventanas
    - devuelve pred diario y lo colapsamos a mensual
    """
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation="relu",
                 random_state=42,
                 max_iter=500):
        self.mlp_params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            random_state=random_state,
            max_iter=max_iter,
        )
        # para guardar todo como en los otros wrappers
        self.model = None
        self.scaler = None      # scaler del asinh
        self.s = None           # escala robusta
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None
        self.colname = None

    def train_from_series(self,
                          x,
                          train_start=dt.datetime(2013, 1, 1),
                          train_end=dt.datetime(2024, 1, 1),
                          colname=None):
        """
        Versión para tu for: recibe la serie y devuelve mensual.
        Por dentro llama a .train(...) que devuelve diario.
        """
        res = self.train(
            x,
            colname=colname,
            test_start=dt.datetime(2024, 2, 1),
            test_end=dt.datetime(2025, 6, 1),
        )

        # pasar diario -> mensual
        pred_daily = pd.Series(np.asarray(res["y_pred"]).ravel(),
                               index=self.test_df.index)
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

    def train(self,
              x,
              colname=None,
              test_start=dt.datetime(2024, 2, 1),
              test_end=dt.datetime(2025, 6, 1)):
        # 1) nombre
        if colname is None:
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # 2) asegurar (N,1)
        if isinstance(x, pd.DataFrame):
            serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
        else:
            serie = pd.Series(x).astype(float).values.reshape(-1, 1)

        # === MISMA LÓGICA QUE TCN ===
        # asinh + MinMax + split
        train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
            serie,
            train_ratio=0.8
        )

        # ventanas 3D: (N, lookback, 1)
        X_train, y_train, X_test, y_test, prediction_days = makewindows(
            train_data,
            test_data
        )
        self.prediction_days = prediction_days

        # MLP de sklearn espera 2D → aplanamos
        N_tr, L, F = X_train.shape
        X_train_flat = X_train.reshape(N_tr, L * F)

        # 3) entrenar MLP
        mlp = MLPRegressor(**self.mlp_params)
        mlp.fit(X_train_flat, y_train.ravel())
        self.model = mlp

        # 4) preparar tramo de test real con fechas (como TCN)
        model_inputs, test_df, t_start, t_end = putTest_cuenta(
            x,
            prediction_days=prediction_days,
            scaler=scaler,
            s=s,
            test_start=test_start,
            test_end=test_end,
        )

        # construir ventanas de test para MLP
        x_test_seq = []
        for i in range(prediction_days, len(model_inputs)):
            x_test_seq.append(model_inputs[i - prediction_days:i, 0])
        x_test_seq = np.array(x_test_seq)  # (N_test, L)
        # aplanar
        x_test_flat = x_test_seq.reshape(x_test_seq.shape[0], -1)

        # 5) predecir
        preds_scaled = mlp.predict(x_test_flat).reshape(-1, 1)
        preds = inverse_scale(preds_scaled, s, scaler)

        # métrica diaria
        y_true = test_df[["Adj Close"]].values
        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))

        # guardar estado
        self.scaler = scaler
        self.s = s
        self.test_df = test_df
        self.test_start = t_start
        self.test_end = t_end
        self.all_preds = [preds.squeeze()]   # para ser igual que TCN/LSTM

        return {
            "y_true": y_true.squeeze(),
            "y_pred": preds.squeeze(),
            "MSE": float(mse),
            "RMSE": rmse,
        }

    def evaluate(self, model=None, X_test=None, y_test=None):
        # devolver mensual como los demás
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) o .train_from_series(...)")

        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(),
                               index=self.test_df.index)
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

    def predecir_futuro(self, *args, **kwargs):
        # igual que en TCN/LSTM: devolvemos lo mensual que ya tenemos
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Entrena primero el MLPSeriesWrapper.")
        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(),
                               index=self.test_df.index)
        pred_month = pred_daily.resample("MS").mean()
        return pred_month.values