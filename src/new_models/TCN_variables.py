#jose antonio st

import datetime as dt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from inspect import isfunction
from src.utils import graficar_x_pred_mensual,escalar_asinh_vector,makewindows,putTest_cuenta,predictionTest,alinear_por_indice

# --- helpers para la misma normalización que usaste al entrenar ---
def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s


def modeloTCN(input_shape):
    """
    input_shape = (lookback, n_features)
    """
    try:
        from tcn import TCN
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
        from tcn import TCN

    model = Sequential()
    model.add(
        TCN(
            nb_filters=64,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16, 32],
            padding="causal",
            dropout_rate=0.2,
            return_sequences=False,
            use_skip_connections=True,
            input_shape=input_shape,
        )
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mae")
    return model


class TCNWrapper:
    """
    Igual interfaz que la tuya, pero:
    - si le pasas 1 columna → usa tu flujo de siempre (asinh + makewindows)
    - si le pasas varias columnas → arma ventanas multivariables aquí mismo
    """
    def __init__(self, lookback=120):
        self.model = None
        self.scaler = None        # scaler del target (asinh)
        self.s = None             # escala del asinh
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None
        self.colname = None

        # para el caso multivar
        self.lookback = lookback
        self.feat_scaler = None   # opcional: escalar variables exógenas

        self.default_ventana = 3
        self.default_flag_ventana = True

    # ------------------------------------------------------------------
    def train_from_series(
        self,
        x: pd.DataFrame | pd.Series,
        train_start=dt.datetime(2013, 1, 1),
        train_end=dt.datetime(2024, 1, 1),
        colname=None,
    ):
        """
        x puede ser:
          - Serie/DataFrame UNA columna (cuenta diaria aumentada)
          - DataFrame varias columnas (cuenta diaria aumentada + variables diarias alineadas)

        Devuelve y_true / y_pred ya EN MENSUAL como tenías.
        """
        # normalizar a DataFrame
        if isinstance(x, pd.Series):
            x = x.to_frame(name=colname or "cuenta")

        if colname is None:
            colname = x.columns[0]
        self.colname = colname

        # CASO 1: UNIVARIADO → tu camino viejo
        if x.shape[1] == 1:
            results = self._train_univariate(
                x,
                colname=colname,
                test_start=dt.datetime(2024, 2, 1),
                test_end=dt.datetime(2025, 6, 1),
            )

            # diario → mensual
            pred_daily = pd.Series(
                np.asarray(results["y_pred"]).ravel(),
                index=self.test_df.index,
            )
            pred_month = pred_daily.resample("MS").mean()

            out = {
                "y_true": self.test_df["Adj Close"].resample("MS").mean().values,
                "y_pred": pred_month.values,
                "MSE": results["MSE"],
                "RMSE": results["RMSE"],
                "idx": pred_month.index,  # para tu plot/export
            }
            return out

        # CASO 2: MULTIVARIADO → nuevo camino
        results = self._train_multivariate(
            x,
            target_col=colname,
            train_start=train_start,
            train_end=train_end,
            test_start=dt.datetime(2024, 2, 1),
            test_end=dt.datetime(2025, 6, 1),
        )

        # results ya viene mensual en este camino
        return results

    # ------------------------------------------------------------------
    #                    C A S O   1   (univar)
    # ------------------------------------------------------------------
    def _train_univariate(
        self,
        x: pd.DataFrame,
        colname=None,
        test_start=dt.datetime(2024, 2, 1),
        test_end=dt.datetime(2025, 6, 1),
    ):
        """
        Es básicamente tu train(...) viejo tal cual.
        """
        # a 1D
        serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)

        # tu escalado
        train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
            serie,
            train_ratio=0.8,
        )

        # tus ventanas
        X_train, y_train, X_test, y_test, prediction_days = makewindows(
            train_data, test_data
        )

        # modelo
        model = modeloTCN((X_train.shape[1], X_train.shape[2]))

        # preparar test con fechas reales
        model_inputs, test_df, t_start, t_end = putTest_cuenta(
            x,
            prediction_days=prediction_days,
            scaler=scaler,
            s=s,
            test_start=test_start,
            test_end=test_end,
        )

        x1_test, _ = predictionTest(
            prediction_days,
            model_inputs,
            model,
            scaler,
            s,
        )

        # entrenar varias veces
        mse_list, rmse_list, all_preds = [], [], []
        for _ in range(10):
            cb = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]
            model.fit(
                X_train,
                y_train,
                epochs=30,
                batch_size=32,
                shuffle=False,
                validation_split=0.1,
                callbacks=cb,
                verbose=0,
                rebuild_each_run=False
            )
            preds_scaled = model.predict(x1_test, verbose=0)
            preds = inverse_scale(preds_scaled, s, scaler)
            y_true = test_df[["Adj Close"]].values
            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse))
            mse_list.append(float(mse))
            rmse_list.append(rmse)
            all_preds.append(preds.squeeze())

        avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
        avg_mse = float(np.mean(mse_list))
        avg_rmse = float(np.mean(rmse_list))

        # guardar
        self.model = model
        self.scaler = scaler
        self.s = s
        self.prediction_days = prediction_days
        self.test_df = test_df
        self.test_start = t_start
        self.test_end = t_end
        self.all_preds = all_preds

        return {
            "y_true": test_df["Adj Close"].values,
            "y_pred": avg_pred,
            "MSE": avg_mse,
            "RMSE": avg_rmse,
        }

    # ------------------------------------------------------------------
    #                    C A S O   2   (multivar)
    # ------------------------------------------------------------------
    def _train_multivariate(
        self,
        df_diario: pd.DataFrame,
        target_col: str,
        train_start: dt.datetime,
        train_end: dt.datetime,
        test_start: dt.datetime,
        test_end: dt.datetime,
    ):
        """
        df_diario: cuenta diaria + variables diarias ya alineadas (lo que armaste con concat)
        target_col: nombre de la cuenta (col) que quieres predecir
        """

        # aseguramos datetimeindex y orden
        if not isinstance(df_diario.index, pd.DatetimeIndex):
            df_diario.index = pd.to_datetime(df_diario.index)
        df_diario = df_diario.sort_index()

        # split por fecha
        df_train = df_diario.loc[:train_end].copy()
        df_test  = df_diario.loc[test_start:test_end].copy()

        # --- escalar SOLO el target con asinh, las exógenas con StandardScaler ---
        y_train_raw = df_train[target_col].values.reshape(-1, 1)

        # tu asinh
        y_train_asinh = np.arcsinh(y_train_raw / 1.0)  # s=1.0 por defecto
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_asinh)

        # features (todas las columnas)
        X_train_raw = df_train.values   # (N, n_features)
        X_test_raw  = df_test.values

        # escalamos features con otro scaler
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train_raw)
        X_test_scaled  = X_scaler.transform(X_test_raw)

        # --- construir ventanas (N_samples, lookback, n_features) ---
        def make_seq(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i - lookback:i, :])
                ys.append(y[i, 0])
            return np.array(Xs, float), np.array(ys, float).reshape(-1, 1)

        # TRAIN (igual que ya tenías)
        X_train_seq, y_train_seq = make_seq(X_train_scaled, y_train_scaled, self.lookback)

        # TEST: cruzar frontera usando el "puente" del final del train
        # 1) puente = últimos lookback del train + TODO el test
        X_bridge = np.vstack([X_train_scaled[-self.lookback:], X_test_scaled])

        # 2) generar exactamente len(test) ventanas empezando en el PRIMER día del test
        X_test_seq = []
        for i in range(self.lookback, self.lookback + len(X_test_scaled)):
            X_test_seq.append(X_bridge[i - self.lookback:i, :])
        X_test_seq = np.array(X_test_seq, dtype="float32")

        # 3) el índice de predicción arranca desde el primer día del test
        test_idx_valid = df_test.index
        # nota: para test estoy tomando la parte final del y_train_scaled solo para tener forma;
        # si quieres test “real” tendrías que aplicar mismo asinh+scaler al y real del test.

        # modelo
        model = modeloTCN((self.lookback, df_diario.shape[1]))

        cb = [
            EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        model.fit(
            X_train_seq,
            y_train_seq,
            epochs=30,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
            callbacks=cb,
            verbose=0,
        )

        # --- predecimos en test (seq) ---
        preds_scaled = model.predict(X_test_seq, verbose=0)   # (len(test), 1) en espacio asinh+std
        # desescalar target: primero des-std, luego sinh
        preds_asinh = y_scaler.inverse_transform(preds_scaled)
        preds = np.sinh(preds_asinh) * 1.0  # s=1.0 si así lo definiste arriba

        # índice de test: TODO el tramo de test (sin recortar)
        test_idx_valid = df_test.index

        # series alineadas 1:1 en DIARIO
        pred_series = pd.Series(preds.ravel(), index=test_idx_valid, name="pred")
        real_series = df_test[target_col]

        # métricas diarias (opcional)
        mse  = mean_squared_error(real_series.values, pred_series.values)
        rmse = float(np.sqrt(mse))

        # diario → mensual
        pred_month = pred_series.resample("MS").mean()
        real_month = real_series.resample("MS").mean()
        common = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[common]
        real_month = real_month.loc[common]

        # guardar en el wrapper (para tu pipeline/plots)
        self.model       = model
        self.scaler      = y_scaler
        self.s           = 1.0
        self.test_df     = real_series.to_frame(name="Adj Close")
        self.test_start  = test_start
        self.test_end    = test_end
        self.all_preds   = [pred_series.values]
        self.feat_scaler = X_scaler

        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": float(mse),
            "RMSE": rmse,
            "idx": pred_month.index,   # por si tu plot usa idx del modelo
        }
