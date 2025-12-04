import datetime as dt
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.utils import (
    escalar_asinh_vector,   # univar
    makewindows,            # univar -> 3D
    putTest_cuenta,         # univar (diario con fechas)
    predictionTest,         # univar test windows
)

# --- helpers univariados (mantener compatibilidad) ---
def forward_scale(x_raw, s, scaler):
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s


class MLPSeriesWrapper:
    """
    MLP para series:
    - Univariado: usa tu pipeline previo (asinh + MinMax + ventanas + aplanado).
    - Multivariable: MinMax por columnas, ventanas 3D -> aplanado 2D, split por fecha,
      desescala sólo la columna target.
    """
    def __init__(self,
                 hidden_layer_sizes=(120, 60),
                 activation="relu",
                 random_state=42,
                 max_iter=800,
                 lookback=120):
        self.mlp_params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            random_state=random_state,
            max_iter=max_iter,
        )
        self.lookback = lookback

        # Estado
        self.model = None
        self.scaler = None      # univar: MinMax de asinh; multivar: MinMax global
        self.s = None           # sólo univar
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None
        self.colname = None
        self.is_multivar = False
        self.n_features_ = 1    # set en multivar

    # ---------------------------
    # API para tu loop principal
    # ---------------------------
    def train_from_series(self,
                          x,
                          train_start=dt.datetime(2013, 1, 1),
                          train_end=dt.datetime(2024, 1, 1),
                          colname=None,
                          test_start=dt.datetime(2024, 2, 1),
                          test_end=dt.datetime(2025, 6, 1)):
        """
        Recibe serie/df diario (ya aumentado). Devuelve métricas y predicción mensual.
        """
        res = self.train(
            x,
            colname=colname,
            test_start=test_start,
            test_end=test_end
        )

        # Pasar de diario -> mensual (como en TCN/LSTM)
        pred_daily = pd.Series(
            np.asarray(res["y_pred"]).ravel(),
            index=self.test_df.index[:len(res["y_pred"])]
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

    # ---------------------------
    # Entrenamiento base
    # ---------------------------
    def train(self,
              x,
              colname=None,
              test_start=dt.datetime(2024, 2, 1),
              test_end=dt.datetime(2025, 6, 1)):
        # Nombre
        if colname is None:
            if isinstance(x, pd.DataFrame):
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        if not isinstance(x, (pd.Series, pd.DataFrame)):
            raise ValueError("x debe ser Serie o DataFrame.")

        # ¿Univar o multivar?
        self.is_multivar = isinstance(x, pd.DataFrame) and x.shape[1] > 1

        if not self.is_multivar:
            # =========================
            #   CAMINO UNIVARIADO
            # =========================
            if isinstance(x, pd.DataFrame):
                serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
                idx = x.index
            else:
                serie = pd.Series(x).astype(float).values.reshape(-1, 1)
                idx = x.index if isinstance(x, pd.Series) else None

            # asinh + split
            train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
                serie,
                train_ratio=0.8
            )

            # ventanas 3D y aplanado
            X_train, y_train, X_test, y_test, prediction_days = makewindows(
                train_data,
                test_data
            )
            self.prediction_days = prediction_days

            N_tr, L, F = X_train.shape
            X_train_flat = X_train.reshape(N_tr, L * F)

            # Entrenar MLP
            mlp = MLPRegressor(**self.mlp_params)
            mlp.fit(X_train_flat, y_train.ravel())
            self.model = mlp

            # Preparar tramo de test con fechas reales
            model_inputs, test_df, t_start, t_end = putTest_cuenta(
                x if isinstance(x, pd.DataFrame) else pd.Series(serie.ravel(), index=idx, name=colname),
                prediction_days=prediction_days,
                scaler=scaler,
                s=s,
                test_start=test_start,
                test_end=test_end,
            )

            # Ventanas de test para MLP
            x_test_seq = []
            for i in range(prediction_days, len(model_inputs)):
                x_test_seq.append(model_inputs[i - prediction_days:i, 0])
            x_test_seq = np.array(x_test_seq)  # (N_test, L)
            x_test_flat = x_test_seq.reshape(x_test_seq.shape[0], -1)

            # Predecir
            preds_scaled = mlp.predict(x_test_flat).reshape(-1, 1)
            preds = inverse_scale(preds_scaled, s, scaler)

            # Métrica diaria
            y_true = test_df[["Adj Close"]].values
            n = min(len(y_true), len(preds))
            y_true = y_true[:n]
            preds = preds[:n]

            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse))

            # Guardar estado
            self.scaler = scaler
            self.s = s
            self.test_df = test_df.iloc[:n].copy()
            self.test_start = t_start
            self.test_end = t_end
            self.all_preds = [preds.squeeze()]
            self.n_features_ = 1

            return {
                "y_true": y_true.squeeze(),
                "y_pred": preds.squeeze(),
                "MSE": float(mse),
                "RMSE": rmse,
            }

        else:
            # =========================
            #   CAMINO MULTIVARIABLE
            # =========================
            df = x.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # target: SIEMPRE la primera columna (cuenta)
            target_name = df.columns[0]
            feats = df.values.astype(float)
            self.n_features_ = feats.shape[1]

            # Escalar por columnas
            scaler = MinMaxScaler()
            feats_s = scaler.fit_transform(feats)

            # Ventanas multivariables: 3D (N, lookback, n_feats)
            L = self.lookback
            X_list, y_list, idx_list = [], [], []
            for i in range(L, len(feats_s)):
                X_list.append(feats_s[i - L:i, :])
                y_list.append(feats_s[i, 0])  # predecimos la col target (pos 0)
                idx_list.append(df.index[i])

            X_arr = np.array(X_list, dtype="float32")
            y_arr = np.array(y_list, dtype="float32").reshape(-1, 1)
            fechas = pd.DatetimeIndex(idx_list)

            # Split por fecha (train: < test_start; test: >= test_start)
            mask_train = fechas < test_start
            mask_test  = fechas >= test_start

            if mask_train.sum() == 0 or mask_test.sum() == 0:
                raise ValueError("No hay suficientes datos para train/test en MLP multivariable.")

            X_train = X_arr[mask_train]
            y_train = y_arr[mask_train]
            X_test  = X_arr[mask_test]
            y_test  = y_arr[mask_test]
            test_idx = fechas[mask_test]

            # Aplanar para MLP (2D)
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat  = X_test.reshape(X_test.shape[0], -1)

            # Entrenar
            mlp = MLPRegressor(**self.mlp_params)
            mlp.fit(X_train_flat, y_train.ravel())
            self.model = mlp

            # Predecir y desescalar SOLO la col objetivo
            preds_s = mlp.predict(X_test_flat).reshape(-1, 1)

            # Para inverse_transform, concatenamos “dummy” n_features y luego tomamos col 0
            dummy_pred = np.zeros((len(preds_s), feats.shape[1]))
            dummy_pred[:, 0] = preds_s[:, 0]
            inv_pred = scaler.inverse_transform(dummy_pred)[:, 0]

            # También desescalamos y_true
            dummy_true = np.zeros((len(y_test), feats.shape[1]))
            dummy_true[:, 0] = y_test[:, 0]
            inv_true = scaler.inverse_transform(dummy_true)[:, 0]

            mse = mean_squared_error(inv_true, inv_pred)
            rmse = float(np.sqrt(mse))

            # Para homogeneidad con TCN/LSTM:
            test_df = pd.DataFrame({"Adj Close": inv_true}, index=test_idx)

            # Guardar estado
            self.scaler = scaler
            self.s = None
            self.prediction_days = L
            self.test_df = test_df
            self.test_start = test_start
            self.test_end = test_end
            self.all_preds = [inv_pred]   # lista para keep same interface

            return {
                "y_true": inv_true,
                "y_pred": inv_pred,
                "MSE": float(mse),
                "RMSE": rmse,
                "idx": test_idx,
            }

    # ---------------------------
    # Evaluación (mensual)
    # ---------------------------
    def evaluate(self, model=None, X_test=None, y_test=None):
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) o .train_from_series(...)")

        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index[:len(avg_pred)])

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

    # ---------------------------
    # Futuro (igual que TCN/LSTM)
    # ---------------------------
    def predecir_futuro(self, *args, **kwargs):
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Entrena primero el MLPSeriesWrapper.")
        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index[:len(avg_pred)])
        pred_month = pred_daily.resample("MS").mean()
        return pred_month.values
