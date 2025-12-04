import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from src.utils import escalar_asinh_vector, makewindows, putTest_cuenta, predictionTest

class LSTMWrapper:
    def __init__(self):
        self.model = None
        self.scaler = None          # puede ser MinMaxScaler (multivar) o el de asinh (univar)
        self.s = None               # solo en univar
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None
        self.colname = None
        self.is_multivar = False    # <--- bandera

    def _build_lstm(self, X_train, units=64, lr=1e-3):
        X_train = np.asarray(X_train)
        timesteps = X_train.shape[1]
        n_features = X_train.shape[2]

        model = Sequential()
        model.add(
            LSTM(
                units,
                activation="relu",
                input_shape=(timesteps, n_features)
            )
        )
        model.add(Dense(1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mae"
        )
        return model

    def train_from_series(
        self,
        x,
        train_start=dt.datetime(2013, 1, 1),
        train_end=dt.datetime(2024, 1, 1),
        colname=None,
        test_start=dt.datetime(2024, 2, 1),
        test_end=dt.datetime(2025, 6, 1),
    ):
        # simplemente usamos train y luego pasamos a mensual
        res = self.train(
            x,
            colname=colname,
            test_start=test_start,
            test_end=test_end,
        )

        # si fue multivariable, el res ya viene alineado al test_df
        pred_daily = pd.Series(
            np.asarray(res["y_pred"]).ravel(),
            index=self.test_df.index[:len(res["y_pred"])]
        )
        pred_month = pred_daily.resample("MS").mean()
        real_month = self.test_df["Adj Close"].resample("MS").mean()

        idx = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[idx]
        real_month = real_month.loc[idx]

        mse = float(((real_month.values - pred_month.values) ** 2).mean())
        rmse = float(np.sqrt(mse))

        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": mse,
            "RMSE": rmse,
        }

    def train(
        self,
        x,
        colname=None,
        test_start=dt.datetime(2024, 2, 1),
        test_end=dt.datetime(2025, 6, 1),
        iterations=5,
        epochs=30,
        batch_size=32,
    ):
        # nombre
        if colname is None:
            if isinstance(x, pd.DataFrame):
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # ---------- detectar si es multivariable ----------
        if isinstance(x, pd.DataFrame) and x.shape[1] > 1:
            self.is_multivar = True
        else:
            self.is_multivar = False

        if not isinstance(x, (pd.Series, pd.DataFrame)):
            raise ValueError("x debe ser Serie o DataFrame.")

        # ====================================================
        # CASO 1: UNIVARIABLE (lo que ya tenías)
        # ====================================================
        if not self.is_multivar:
            if isinstance(x, pd.DataFrame):
                serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
            else:
                serie = pd.Series(x).astype(float).values.reshape(-1, 1)

            # asinh + split
            train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
                serie,
                train_ratio=0.8
            )

            X_train, y_train, X_test, y_test, prediction_days = makewindows(
                train_data,
                test_data
            )

            model = self._build_lstm(X_train, units=64, lr=1e-3)

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
                s
            )

            mse_list, rmse_list, all_preds = [], [], []
            for _ in range(iterations):
                m_i = self._build_lstm(X_train, units=64, lr=1e-3)
                cb = [tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0
                )]
                m_i.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    validation_split=0.1,
                    callbacks=cb,
                    verbose=0
                )

                preds_scaled = m_i.predict(x1_test, verbose=0)
                preds = inverse_scale(preds_scaled, s, scaler)

                y_true = test_df[["Adj Close"]].values
                n = min(len(y_true), len(preds))
                y_true = y_true[:n]
                preds = preds[:n]

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
                "y_true": test_df["Adj Close"].values[:len(avg_pred)],
                "y_pred": avg_pred,
                "MSE": avg_mse,
                "RMSE": avg_rmse,
            }

        # ====================================================
        # CASO 2: MULTIVARIABLE
        # ====================================================
        # x es DataFrame con 1 (cuenta) + k variables
        df = x.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # target es SIEMPRE la primera col (la cuenta aumentada)
        target = df.iloc[:, 0].to_frame("Adj Close")
        feats = df.values.astype(float)

        # escalamos todas las columnas con MinMax
        scaler = MinMaxScaler()
        feats_s = scaler.fit_transform(feats)

        # hacemos ventanas manuales: igual que makewindows pero multivar
        lookback = 120
        X_list, y_list = [], []
        for i in range(lookback, len(feats_s)):
            X_list.append(feats_s[i - lookback:i, :])   # (lookback, n_feats)
            y_list.append(feats_s[i, 0])                 # predecimos la primera columna
        X_arr = np.array(X_list, dtype="float32")
        y_arr = np.array(y_list, dtype="float32").reshape(-1, 1)

        # split por fecha: si tu aumento está alineado, podemos tomar índice
        fechas = df.index[lookback:]     # alineado a X_arr
        mask_train = fechas < test_start
        mask_test  = fechas >= test_start

        X_train = X_arr[mask_train]
        y_train = y_arr[mask_train]
        X_test  = X_arr[mask_test]
        y_test  = y_arr[mask_test]
        test_idx = fechas[mask_test]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("No hay suficientes datos para train/test en LSTM multivariable.")

        model = self._build_lstm(X_train, units=64, lr=1e-3)

        mse_list, rmse_list, all_preds = [], [], []
        for _ in range(iterations):
            m_i = self._build_lstm(X_train, units=64, lr=1e-3)
            cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
                verbose=0
            )]
            m_i.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                validation_split=0.1,
                callbacks=cb,
                verbose=0
            )

            preds_s = m_i.predict(X_test, verbose=0)
            # solo desescalamos 1 columna (la target). Como usamos MinMax global,
            # podemos hacer inverse_transform concatenando.
            # armamos un array vacío para inverse_transform
            dummy = np.zeros((len(preds_s), feats.shape[1]))
            dummy[:, 0] = preds_s[:, 0]
            inv = scaler.inverse_transform(dummy)
            preds = inv[:, 0]

            # real también lo desescalamos
            dummy_true = np.zeros((len(y_test), feats.shape[1]))
            dummy_true[:, 0] = y_test[:, 0]
            inv_true = scaler.inverse_transform(dummy_true)
            y_true = inv_true[:, 0]

            mse = mean_squared_error(y_true, preds)
            rmse = float(np.sqrt(mse))
            mse_list.append(float(mse))
            rmse_list.append(rmse)
            all_preds.append(preds.squeeze())

        avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)

        # guardamos como si fuera test_df
        test_df = pd.DataFrame({
            "Adj Close": y_true
        }, index=test_idx)

        self.model = model
        self.scaler = scaler
        self.s = None
        self.prediction_days = lookback
        self.test_df = test_df
        self.test_start = test_start
        self.test_end = test_end
        self.all_preds = all_preds

        return {
            "y_true": y_true,
            "y_pred": avg_pred,
            "MSE": float(np.mean(mse_list)),
            "RMSE": float(np.mean(rmse_list)),
            "idx": test_idx,
        }
