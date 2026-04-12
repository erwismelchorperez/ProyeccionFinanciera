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
def _infer_step_days(idx: pd.DatetimeIndex, default=2):
    # intenta inferir el paso típico en días (p.ej. 2 si aumentaste días impares)
    if len(idx) < 3:
        return default
    diffs = pd.Series(idx[1:] - idx[:-1]).dt.days
    mode = diffs.mode()
    return int(mode.iloc[0]) if not mode.empty else default

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
                      colname=None,
                      test_start=dt.datetime(2024, 2, 1),
                      test_end=dt.datetime(2025, 6, 1)):
        """
        Versión para tu for: recibe la serie y devuelve mensual.
        Por dentro llama a .train(...) que devuelve diario.
        """
        res = self.train(
            x,
            colname=colname,
            test_start=test_start,
            test_end=test_end,
        )

        # pasar diario -> mensual (esto igual que lo tenías)
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

    def predecir_futuro(self,
                        x_diaria: pd.DataFrame,
                        start_forecast: pd.Timestamp,
                        meses_a_predecir: int = 12,
                        ventana: int = None,
                        flag_ventana: bool = True):
        """
        Forecast autoregresivo mensual usando el MLP de sklearn.
        Fuerza 1D en todo el flujo para evitar 'Data must be 1-dimensional'.
        """
        if self.model is None or self.scaler is None or self.s is None:
            raise ValueError("MLPSeriesWrapper no entrenado. Llama antes a train_from_series.")

        # --- Serie (una sola columna), ordenada y con índice datetime ---
        serie = x_diaria.iloc[:, 0].astype(float)
        if not isinstance(serie.index, pd.DatetimeIndex):
            serie.index = pd.to_datetime(serie.index)
        serie = serie.sort_index()

        # Historial hasta el día previo al arranque del forecast
        corte = pd.to_datetime(start_forecast) - pd.Timedelta(days=1)
        serie_hist = serie.loc[:corte]

        # Helpers de escalado → siempre devolver 2D y luego ravel donde toque
        def scale_vals(vals: np.ndarray) -> np.ndarray:
            arr = np.asarray(vals, dtype=float).reshape(-1, 1)
            arr_asinh = np.arcsinh(arr / self.s)
            return self.scaler.transform(arr_asinh)                 # (N, 1)

        def inverse_vals(vals_scaled_1d: np.ndarray) -> np.ndarray:
            arr = np.asarray(vals_scaled_1d, dtype=float).reshape(-1, 1)  # (N,1)
            arr_asinh = self.scaler.inverse_transform(arr)                # (N,1)
            return (np.sinh(arr_asinh) * self.s).ravel()                  # (N,)

        # Escalar historial y armar ventana
        hist_scaled = scale_vals(serie_hist.values)   # (N_hist, 1)
        lookback = int(self.prediction_days)
        if len(hist_scaled) < lookback:
            raise ValueError("Serie histórica muy corta para la ventana del MLP.")

        # ventana 1D para sklearn (1, lookback)
        window = hist_scaled[-lookback:, 0].copy().reshape(1, lookback)

        # Inferir paso diario del aumento (p.ej. 2 días)
        step_days = _infer_step_days(serie.index, default=2)

        # Construir un índice “diario aumentado” desde start_forecast
        # Usamos 31 días por mes para cubrir todos los casos y recortamos luego a meses_a_predecir
        pasos_dias = int(meses_a_predecir * (31 / step_days))
        idx_daily = pd.date_range(
            start=pd.to_datetime(start_forecast),
            periods=pasos_dias,
            freq=f"{step_days}D"
        )

        # Autoregresivo en espacio escalado
        preds_scaled_list = []
        for _ in range(len(idx_daily)):
            y_scaled_1d = self.model.predict(window)      # shape (1,)
            y_scalar = float(y_scaled_1d[0])              # escalar puro
            preds_scaled_list.append(y_scalar)

            # shift ventana
            window = np.roll(window, shift=-1, axis=1)
            window[0, -1] = y_scalar

        # Invertir escala → 1D
        preds_original_1d = inverse_vals(np.array(preds_scaled_list, dtype=float))  # (N,)

        # Serie diaria y colapso a mensual (MS)
        serie_future_daily = pd.Series(preds_original_1d, index=idx_daily)  # <-- valores 1D
        serie_future_month = serie_future_daily.resample("MS").mean().iloc[:meses_a_predecir]

        return {
            "idx": serie_future_month.index,
            "y_pred": serie_future_month.values  # (M,) 1D
        }


def make_monthly_windows(y_m, lookback=12):
    X, y = [], []
    for i in range(lookback, len(y_m)):
        X.append(y_m[i-lookback:i])   # (lookback,)
        y.append(y_m[i])              # (1,)
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    return X, y

class MLPMonthlyWrapper:
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 random_state=42, max_iter=500, lookback=12):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                  activation=activation,
                                  random_state=random_state,
                                  max_iter=max_iter)
        self.lookback = lookback
        self.scaler_y = StandardScaler()   # asinh+std como en TCN/LSTM
        self.s = 1.0

    def train_from_series(self, x_diaria, train_start, train_end, colname=None,
                          test_start=None, test_end=None):
        # 1) pasar a mensual
        y_m = x_diaria.iloc[:,0].asfreq("D").resample("MS").mean().astype(float)
        y_m = y_m.sort_index()

        y_train = y_m.loc[:train_end].values.reshape(-1,1)
        y_asinh = np.arcsinh(y_train/self.s)
        y_tr_s  = self.scaler_y.fit_transform(y_asinh).ravel()

        # 2) ventanas
        X_tr, y_tr = make_monthly_windows(y_tr_s, lookback=self.lookback)

        # 3) entrenar
        self.model.fit(X_tr, y_tr.ravel())

        # 4) test mensual “real”
        y_test_m = y_m.loc[test_start:test_end]
        # construir sus ventanas usando la misma normalización
        full_asinh = np.arcsinh(y_m.values.reshape(-1,1)/self.s)
        full_s     = self.scaler_y.transform(full_asinh).ravel()
        # idx del primer punto de test que tiene lookback
        pos0 = y_m.index.get_loc(test_start)
        X_te = []
        for i in range(pos0, pos0 + len(y_test_m)):
            if i < self.lookback: continue
            X_te.append(full_s[i-self.lookback:i])
        X_te = np.array(X_te)
        y_pred_s = self.model.predict(X_te)
        y_pred   = (np.sinh(self.scaler_y.inverse_transform(y_pred_s.reshape(-1,1))) * self.s).ravel()

        idx = y_test_m.index[:len(y_pred)]
        y_true = y_test_m.values[:len(y_pred)]
        mse  = float(np.mean((y_true - y_pred)**2))
        rmse = float(np.sqrt(mse))

        return {"y_true": y_true, "y_pred": y_pred, "MSE": mse, "RMSE": rmse, "idx": idx}

    def predecir_futuro(self, x_mensual, start_forecast, meses_a_predecir):
        # x_mensual: Serie mensual (MS) completa hasta el mes anterior a start_forecast
        y_m = x_mensual.asfreq("MS").astype(float).sort_index()
        y_hist = y_m.loc[: start_forecast - pd.offsets.MonthBegin(0)].values.reshape(-1,1)

        full_asinh = np.arcsinh(y_hist/self.s)
        full_s     = self.scaler_y.transform(full_asinh).ravel()

        if len(full_s) < self.lookback:
            raise ValueError("Hist mensual insuficiente para la ventana.")

        window = full_s[-self.lookback:].copy()  # (L,)
        preds = []
        for _ in range(meses_a_predecir):
            yhat_s = self.model.predict(window.reshape(1,-1))[0]
            preds.append(yhat_s)
            window = np.roll(window, -1)
            window[-1] = yhat_s

        yhat = (np.sinh(self.scaler_y.inverse_transform(np.array(preds).reshape(-1,1))) * self.s).ravel()
        idx  = pd.date_range(start=start_forecast, periods=meses_a_predecir, freq="MS")
        return {"idx": idx, "y_pred": yhat}
