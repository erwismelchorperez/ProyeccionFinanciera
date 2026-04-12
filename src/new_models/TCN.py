import datetime as dt
import numpy as np
import pandas as pd
from src.utils import graficar_x_pred_mensual,escalar_asinh_vector,makewindows,putTest_cuenta,predictionTest,alinear_por_indice
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import sys,subprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from inspect import isfunction
from sklearn.metrics import mean_squared_error


# --- helpers para la misma normalización que usaste al entrenar ---
def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s

def entrenamiento(model_or_fn, X_train, y_train, x1_test, scaler, s, test_df,
                  iterations=10, epochs=30, batch_size=32, rebuild_each_run=True,
                  verbose=0, use_val_split=True, patience=8):
    mse_list, rmse_list, all_preds = [], [], []

    for i in range(iterations):
        if rebuild_each_run and isfunction(model_or_fn):
            model = model_or_fn(X_train)     # construir nuevo modelo
        else:
            model = model_or_fn              # usar la instancia ya creada

        callbacks = []
        if use_val_split:
            callbacks = [EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True, verbose=0)]

        model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            shuffle=False,
            validation_split=0.1 if use_val_split else 0.0,
            callbacks=callbacks, verbose=verbose
        )

        preds_scaled = model.predict(x1_test, verbose=0)
        preds = inverse_scale(preds_scaled, s, scaler)

        y_true = test_df[['Adj Close']].values
        mse  = mean_squared_error(y_true, preds)
        #y_true_al, y_pred_al = alinear_por_indice(test_df, preds)
        #mse  = mean_squared_error(y_true_al, y_pred_al)

        rmse = float(np.sqrt(mse))

        mse_list.append(float(mse))
        rmse_list.append(rmse)
        all_preds.append(preds.squeeze())

    avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    avg_mse  = float(np.mean(mse_list))
    avg_rmse = float(np.mean(rmse_list))
    return avg_pred, avg_mse, avg_rmse, all_preds, mse_list, rmse_list, iterations

def modeloTCN(X_train):
    # 0) Validaciones rápidas
    X_train = np.asarray(X_train)
    if X_train.ndim != 3:
        raise ValueError(f"X_train debe ser (N, lookback, n_features); recibido {X_train.shape}")
    if X_train.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 muestras para entrenar.")
    X_train = X_train.astype("float32", copy=False)

    # 1) Importar TCN (instalar si falta)
    try:
        from tcn import TCN
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tcn"])
        from tcn import TCN

    # 2) Definir el modelo
    lookback  = X_train.shape[1]
    n_features = X_train.shape[2]

    # Nota: con kernel_size=3 y dilations=[1,2,4,8,16,32] tu lookback>=~96 va bien (tú usas 120)
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
            input_shape=(lookback, n_features),
        )
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mae")
    print("X_train shape:", X_train.shape)
    return model


class TCNWrapper:
    def __init__(self):
        # lo que ya tenías
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
    # 1) API "nueva" para tu loop: recibe la serie (diaria o mensual) y
    #    las fechas de entrenamiento
    # ------------------------------------------------------------------
    def train_from_series(self,
                      x,
                      train_start=dt.datetime(2013, 1, 1),
                      train_end=dt.datetime(2024, 1, 1),
                      colname=None,
                      test_start=dt.datetime(2024, 2, 1),
                      test_end=dt.datetime(2025, 6, 1)):

        """
        x: Series/DataFrame de UNA cuenta (puede ya venir diaria porque la aumentaste).
        train_start / train_end: las fechas que tu pipeline quiere usar.
        La lógica interna sigue siendo la misma que tu .train(...) anterior,
        solo que guardamos mejor los resultados y devolvemos mensual.
        """
        #guardamos nombre
        if colname is None:
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # llamamos a train usando el rango de test que venga de fuera
        results = self.train(
            x,
            colname=colname,
            test_start=test_start,
            test_end=test_end,
        )

        # results viene con diario en y_pred; lo convertimos a mensual aquí
        pred_daily = pd.Series(
            np.asarray(results["y_pred"]).ravel(),
            index=self.test_df.index
        )
        pred_month = pred_daily.resample("MS").mean()

        out = {
            "y_true": self.test_df["Adj Close"].resample("MS").mean().values,
            "y_pred": pred_month.values,
            "MSE": results["MSE"],
            "RMSE": results["RMSE"],
        }
        return out

    # ------------------------------------------------------------------
    # 2) tu train original, pero le permito pasar test_start/test_end
    # ------------------------------------------------------------------
    def train(self, x, colname=None,
              test_start=dt.datetime(2024, 2, 1),
              test_end=dt.datetime(2025, 6, 1)):
        """
        Tu lógica tal cual, solo que test_start/test_end vienen por parámetro.
        """
        # 1) nombre
        if colname is None:
            if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
                colname = x.columns[0]
            else:
                colname = "cuenta"
        self.colname = colname

        # 2) asegurar serie 1D -> (N,1)
        if isinstance(x, pd.DataFrame):
            serie = x.iloc[:, 0].astype(float).values.reshape(-1, 1)
        else:
            serie = pd.Series(x).astype(float).values.reshape(-1, 1)

        # 3) asinh + split (lo tuyo)
        train_data, test_data, scaler, s, n_train = escalar_asinh_vector(
            serie,
            train_ratio=0.8
        )

        # 4) ventanas
        X_train, y_train, X_test, y_test, prediction_days = makewindows(
            train_data, test_data
        )

        # 5) modelo
        model = modeloTCN(X_train)

        # 6) preparar test con fechas reales de la cuenta
        model_inputs, test_df, t_start, t_end = putTest_cuenta(
            x,                      # serie con índice datetime
            prediction_days=prediction_days,
            scaler=scaler,
            s=s,
            test_start=test_start,
            test_end=test_end,
        )

        # 7) inputs de test para keras
        x1_test, _ = predictionTest(
            prediction_days,
            model_inputs,
            model,
            scaler,
            s
        )

        # 8) entrenamiento repetido (lo tuyo)
        avg_pred, avg_mse, avg_rmse, all_preds, mse_list, rmse_list, iters = entrenamiento(
            model_or_fn=model,
            X_train=X_train, y_train=y_train,
            x1_test=x1_test,
            scaler=scaler, s=s,
            test_df=test_df,
            iterations=10,
            epochs=30, batch_size=32,
            rebuild_each_run=True
        )

        # 9) guardar en el wrapper
        self.model = model
        self.scaler = scaler
        self.s = s
        self.prediction_days = prediction_days
        self.test_df = test_df
        self.test_start = t_start
        self.test_end = t_end
        self.all_preds = all_preds

        # 10) devolvemos como antes (DIARIO)
        return {
            "y_true": test_df["Adj Close"].values,
            "y_pred": avg_pred,
            "MSE": float(avg_mse),
            "RMSE": float(avg_rmse),
        }

    # ------------------------------------------------------------------
    def evaluate(self, model=None, X_test=None, y_test=None):
        """
        Ahora devolvemos MENSUAL por defecto, porque tu serie base es mensual.
        """
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Primero llama a .train(...) o .train_from_series(...)")

        # promedio diario
        avg_pred = np.mean(np.stack(self.all_preds, axis=0), axis=0)
        pred_daily = pd.Series(np.asarray(avg_pred).ravel(), index=self.test_df.index)

        # diario → mensual
        pred_month = pred_daily.resample("MS").mean()
        real_month = self.test_df["Adj Close"].resample("MS").mean()

        # alineamos
        common_idx = pred_month.index.intersection(real_month.index)
        pred_month = pred_month.loc[common_idx]
        real_month = real_month.loc[common_idx]

        mse = float(((real_month.values - pred_month.values) ** 2).mean())
        rmse = float(np.sqrt(mse))
        return {
            "y_true": real_month.values,
            "y_pred": pred_month.values,
            "MSE": mse,
            "RMSE": rmse
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
    # ------------------------------------------------------------------
    def predecir_futuro(self,
                        x_diaria: pd.DataFrame,
                        start_forecast: pd.Timestamp,
                        meses_a_predecir: int = 12,
                        ventana: int = None,
                        flag_ventana: bool = True):
        """
        Forecast autoregresivo en DIARIO usando el mismo scaling del entrenamiento
        y luego se colapsa a MENSUAL (MS).

        x_diaria: serie diaria aumentada (la misma que pasaste a train_from_series).
        start_forecast: fecha a partir de la cual quieres predecir (ej. 2024-04-01).
        meses_a_predecir: número de meses futuros (ej. 57).
        """
        if self.model is None or self.scaler is None or self.s is None:
            raise ValueError("TCNWrapper no entrenado. Llama antes a train_from_series.")

        # para invertir escala
        from src.utils import inverse_scale

        # 1) asegurar índice datetime
        serie = x_diaria.iloc[:, 0].astype(float)
        if not isinstance(serie.index, pd.DatetimeIndex):
            serie.index = pd.to_datetime(serie.index)
        serie = serie.sort_index()

        # 2) histórico hasta el día antes del forecast
        serie_hist = serie.loc[: start_forecast - pd.Timedelta(days=1)]

        # 3) mismas funciones de escala que en entrenamiento
        def scale_vals(vals: np.ndarray) -> np.ndarray:
            arr = vals.reshape(-1, 1)
            arr_asinh = np.arcsinh(arr / self.s)
            arr_scaled = self.scaler.transform(arr_asinh)
            return arr_scaled

        def inverse_vals(vals_scaled: np.ndarray) -> np.ndarray:
            return inverse_scale(vals_scaled.reshape(-1, 1), self.s, self.scaler).ravel()

        # 4) escalar histórico y preparar ventana
        hist_scaled = scale_vals(serie_hist.values)
        lookback = self.prediction_days  # p.ej. 120

        if len(hist_scaled) < lookback:
            raise ValueError("Serie histórica muy corta para la ventana del TCN.")

        # ventana (1, lookback, 1) para Keras
        window = hist_scaled[-lookback:].copy().reshape(1, lookback, 1)

        # 5) bucle autoregresivo en DIARIO
        dias_por_mes_aprox = 21
        pasos_dias = meses_a_predecir * dias_por_mes_aprox

        preds_scaled = []
        for _ in range(pasos_dias):
            y_scaled = self.model.predict(window, verbose=0)  # shape (1,1)
            preds_scaled.append(y_scaled[0, 0])

            # shift ventana
            window = np.roll(window, shift=-1, axis=1)
            window[0, -1, 0] = y_scaled[0, 0]

        preds_scaled_arr = np.array(preds_scaled)
        preds_original = inverse_vals(preds_scaled_arr)

        # 6) índice diario futuro
        idx_daily = pd.date_range(
            start=serie_hist.index[-1] + pd.Timedelta(days=1),
            periods=pasos_dias,
            freq="D"
        )

        serie_future_daily = pd.Series(preds_original, index=idx_daily)

        # 7) mensualizar y recortar a meses_a_predecir
        serie_future_month = serie_future_daily.resample("MS").mean().iloc[:meses_a_predecir]

        return {
            "idx": serie_future_month.index,     # DatetimeIndex mensual
            "y_pred": serie_future_month.values  # np.array
        }

    # ------------------------------------------------------------------
    def plot(self, x, end_limit="2025-06-01"):
        if self.test_df is None or self.all_preds is None:
            raise ValueError("Entrena primero el TCNWrapper con .train(...)")
        _ = graficar_x_pred_mensual(
            x=x,
            all_preds=self.all_preds,
            test_df=self.test_df,
            test_start=self.test_start,
            end_limit=end_limit,
            titulo=f"Cuenta {self.colname}: Real mensual vs Pred mensual"
        )

