import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from src.utils import escalar_asinh_vector,makewindows,putTest_cuenta,predictionTest,graficar_x_pred_mensual

# makewindows, putTest_cuenta, predictionTest
# --- helpers para la misma normalización que usaste al entrenar ---
def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s


class LSTMWrapper:
    def __init__(self):
        # lo que vamos a ir guardando
        self.model = None
        self.scaler = None
        self.s = None
        self.prediction_days = None
        self.test_df = None
        self.test_start = None
        self.test_end = None
        self.all_preds = None
        self.colname = None

    # ------------------------------------------------------------------
    # construye el modelo LSTM en base a X_train (igual que hicimos con TCN)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # igual que en TCN: una API para tu for → recibe la serie (diaria o mensual)
    # y las fechas de train/test
    # ------------------------------------------------------------------
    def train_from_series(self,
                          x,
                          train_start=dt.datetime(2013, 1, 1),
                          train_end=dt.datetime(2024, 1, 1),
                          colname=None,
                          test_start=dt.datetime(2024, 2, 1),
                          test_end=dt.datetime(2025, 6, 1)):
        """
        x: Serie/DataFrame de UNA cuenta (ya aumentada a diaria).
        La lógica interna es la misma que .train(), solo que devolvemos
        ya en mensual para que tu loop lo pueda guardar igual que otros.
        """
        results = self.train(
            x,
            colname=colname,
            test_start=test_start,
            test_end=test_end
        )

        # convertir diario → mensual
        pred_daily = pd.Series(
            np.asarray(results["y_pred"]).ravel(),
            index=self.test_df.index
        )
        pred_month = pred_daily.resample("MS").mean()
        real_month = self.test_df["Adj Close"].resample("MS").mean()

        # alinear
        idx = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[idx]
        real_month = real_month.loc[idx]

        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": float(((real_month.values - pred_month.values) ** 2).mean()),
            "RMSE": float(np.sqrt(((real_month.values - pred_month.values) ** 2).mean()))
        }

    # ------------------------------------------------------------------
    # tu train “real”, igual que TCN pero con LSTM
    # ------------------------------------------------------------------
    def train(self,
              x,
              colname=None,
              test_start=dt.datetime(2024, 1, 1),
              test_end=dt.datetime(2025, 6, 1),
              iterations=10,
              epochs=30,
              batch_size=32):
        # 1) nombre
        if colname is None:
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # 2) asegurar serie (N,1)
        if isinstance(x, pd.DataFrame):
            serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
        else:
            serie = pd.Series(x).astype(float).values.reshape(-1, 1)

        # 3) igual que TCN → asinh + split
        train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
            serie,
            train_ratio=0.8
        )

        # 4) ventanas para LSTM (usa tu misma makewindows, que ya devuelve 3D)
        X_train, y_train, X_test, y_test, prediction_days = makewindows(
            train_data,
            test_data
        )

        # 5) modelo LSTM
        model = self._build_lstm(X_train, units=64, lr=1e-3)

        # 6) preparar test con fechas reales (igual que TCN)
        model_inputs, test_df, t_start, t_end = putTest_cuenta(
            x,
            prediction_days=prediction_days,
            scaler=scaler,
            s=s,
            test_start=test_start,
            test_end=test_end
        )

        # 7) inputs de test para Keras
        x1_test, _ = predictionTest(
            prediction_days,
            model_inputs,
            model,
            scaler,
            s
        )

        # 8) entrenamiento repetido (mismo patrón que TCN)
        mse_list, rmse_list, all_preds = [], [], []
        for i in range(iterations):
            # re-construimos para que cada iter sea “fresh”
            model_i = self._build_lstm(X_train, units=64, lr=1e-3)
            cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=0
            )]
            model_i.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                validation_split=0.1,
                callbacks=cb,
                verbose=0
            )

            preds_scaled = model_i.predict(x1_test, verbose=0)
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

        # guardamos todo
        self.model = model           # el primero, o podrías guardar el último
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
    def evaluate(self, model=None, X_test=None, y_test=None):
        """
        Igual que en TCN: ya tenemos las predicciones diarias,
        aquí las colapsamos a mensual y devolvemos métricas.
        """
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) o .train_from_series(...)")

        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index)
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


    def forecast_future_meses(self,
                          x_diaria: pd.DataFrame,
                          start_forecast: pd.Timestamp,
                          meses_a_predecir: int):
        """
        Hace un forecast autoregresivo de `meses_a_predecir` meses
        usando la MISMA escala que se usó para entrenar
        (self.scaler, self.s, self.prediction_days).

        x_diaria: serie diaria aumentada (la misma que pasas a train_from_series).
        """
        if self.model is None or self.scaler is None or self.s is None:
            raise ValueError("Wrapper no entrenado. Llama antes a train_from_series.")

        # 1) asegurar índice datetime
        serie = x_diaria.iloc[:, 0].astype(float)
        if not isinstance(serie.index, pd.DatetimeIndex):
            serie.index = pd.to_datetime(serie.index)
        serie = serie.sort_index()

        # 2) tomar solo hasta el último día antes de start_forecast
        serie_hist = serie.loc[: start_forecast - pd.Timedelta(days=1)]

        # 3) función para ESCALAR igual que en el entrenamiento
        #    (ajústala exactamente a tu escalar_asinh_vector / inverse_scale)
        def scale_vals(vals: np.ndarray) -> np.ndarray:
            arr = vals.reshape(-1, 1)
            # mismo asinh que usaste antes
            arr_asinh = np.arcsinh(arr / self.s)
            arr_scaled = self.scaler.transform(arr_asinh)
            return arr_scaled

        def inverse_vals(vals_scaled: np.ndarray) -> np.ndarray:
            # usa tu inverse_scale real si la tienes
            from src.utils import inverse_scale
            return inverse_scale(vals_scaled.reshape(-1, 1), self.s, self.scaler).ravel()

        # 4) escalar TODO el histórico
        hist_scaled = scale_vals(serie_hist.values)
        lookback = self.prediction_days  # p.ej. 120

        # ventana inicial: últimos `lookback` puntos escalados
        if len(hist_scaled) < lookback:
            raise ValueError("Serie histórica muy corta para la ventana del modelo.")
        window = hist_scaled[-lookback:].copy().reshape(1, lookback, 1)

        # 5) bucle autoregresivo diario
        #    (puedes ajustar si quieres trabajar por meses directamente)
        preds_scaled = []
        last_date = serie_hist.index[-1]
        # si tu aumento mensual agrega, p.ej., 21 días por mes:
        dias_por_mes_aprox = 21
        pasos_dias = meses_a_predecir * dias_por_mes_aprox

        for _ in range(pasos_dias):
            # TCN / LSTM usan (1, lookback, 1)
            y_scaled = self.model.predict(window, verbose=0)
            preds_scaled.append(y_scaled[0, 0])

            # corrimiento de ventana: quitamos el primero y añadimos el nuevo
            window = np.roll(window, shift=-1, axis=1)
            window[0, -1, 0] = y_scaled[0, 0]

            last_date = last_date + pd.Timedelta(days=1)

        # 6) desescalar a unidad original
        preds_scaled_arr = np.array(preds_scaled)
        preds_original = inverse_vals(preds_scaled_arr)

        # 7) construir índice diario futuro
        idx_daily = pd.date_range(
            start=serie_hist.index[-1] + pd.Timedelta(days=1),
            periods=pasos_dias,
            freq="D"
        )

        serie_future_daily = pd.Series(preds_original, index=idx_daily)

        # 8) convertir a mensual (MS) y recortar a meses_a_predecir
        serie_future_month = serie_future_daily.resample("MS").mean().iloc[:meses_a_predecir]

        return {
            "idx": serie_future_month.index,
            "y_pred": serie_future_month.values,
        }


    # ------------------------------------------------------------------
    def predecir_futuro(self,
                        x_diaria: pd.DataFrame,
                        start_forecast: pd.Timestamp,
                        meses_a_predecir: int = 12,
                        ventana: int = None,
                        flag_ventana: bool = True):
        """
        Forecast autoregresivo mensual usando el LSTM entrenado.
        """
        if self.model is None or self.scaler is None or self.s is None:
            raise ValueError("LSTMWrapper no entrenado. Llama antes a train_from_series.")

        from src.utils import inverse_scale

        serie = x_diaria.iloc[:, 0].astype(float)
        if not isinstance(serie.index, pd.DatetimeIndex):
            serie.index = pd.to_datetime(serie.index)
        serie = serie.sort_index()

        serie_hist = serie.loc[: start_forecast - pd.Timedelta(days=1)]

        def scale_vals(vals: np.ndarray) -> np.ndarray:
            arr = vals.reshape(-1, 1)
            arr_asinh = np.arcsinh(arr / self.s)
            arr_scaled = self.scaler.transform(arr_asinh)
            return arr_scaled

        def inverse_vals(vals_scaled: np.ndarray) -> np.ndarray:
            return inverse_scale(vals_scaled.reshape(-1, 1), self.s, self.scaler).ravel()

        hist_scaled = scale_vals(serie_hist.values)
        lookback = self.prediction_days

        if len(hist_scaled) < lookback:
            raise ValueError("Serie histórica muy corta para la ventana del LSTM.")

        window = hist_scaled[-lookback:].copy().reshape(1, lookback, 1)

        dias_por_mes_aprox = 21
        pasos_dias = meses_a_predecir * dias_por_mes_aprox

        preds_scaled = []
        for _ in range(pasos_dias):
            y_scaled = self.model.predict(window, verbose=0)
            preds_scaled.append(y_scaled[0, 0])

            window = np.roll(window, shift=-1, axis=1)
            window[0, -1, 0] = y_scaled[0, 0]

        preds_scaled_arr = np.array(preds_scaled)
        preds_original = inverse_vals(preds_scaled_arr)

        idx_daily = pd.date_range(
            start=serie_hist.index[-1] + pd.Timedelta(days=1),
            periods=pasos_dias,
            freq="D"
        )
        serie_future_daily = pd.Series(preds_original, index=idx_daily)
        serie_future_month = serie_future_daily.resample("MS").mean().iloc[:meses_a_predecir]

        return {
            "idx": serie_future_month.index,
            "y_pred": serie_future_month.values
        }


    # ------------------------------------------------------------------
    def plot(self, x, end_limit="2025-06-01"):
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Entrena primero el LSTMWrapper.")
        # si ya tienes esta función la reutilizas; si no, es la misma que usas para TCN
        _ = graficar_x_pred_mensual(
            x=x,
            all_preds=self.all_preds,
            test_df=self.test_df,
            test_start=self.test_start,
            end_limit=end_limit,
            titulo=f"Cuenta {self.colname}: Real mensual vs Pred mensual (LSTM)"
        )
