import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
try:
    from tcn import TCN
except ImportError:
    # Por si la clase está en tcn.tcn en tu versión
    from tcn.tcn import TCN

import keras
# En caso necesario, registramos explícitamente
keras.saving.register_keras_serializable(package="Custom")(TCN)
# ============================================================
_MES_MAP_EN = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def parse_mes_en(tag: str) -> pd.Timestamp:
    """Convierte Mar-15 → Timestamp(2015-03-01)."""
    s = str(tag).strip().lower().replace(" ", "").replace("_", "-")
    m = re.match(r"^([a-z]{3})-(\d{2}|\d{4})$", s)
    if not m:
        raise ValueError(f"Etiqueta de mes no reconocida: '{tag}'")

    m3, y = m.groups()
    if m3 not in _MES_MAP_EN:
        raise ValueError(f"Mes no reconocido: '{m3}' en '{tag}'")

    year = int(y)
    if year < 100:
        year += 2000

    return pd.Timestamp(year=year, month=_MES_MAP_EN[m3], day=1)

# ============================================================
# Escalado
# ============================================================
def scale_window_asinh(window_values, scaler, s):
    asinh_vals = np.arcsinh(window_values / s)
    return scaler.transform(asinh_vals)

def inverse_scale_single(y_scaled, scaler, s):
    asinh_val = scaler.inverse_transform(np.array([[y_scaled]]))[0, 0]
    return float(np.sinh(asinh_val) * s)

# ============================================================
# Forecast autoregresivo
# ============================================================
def forecast_autoregresivo_desde_joblib(model_path, serie_real, start_date, horizon):
    wrapper = joblib.load(model_path)

    serie_real = serie_real.sort_index().asfreq("MS")
    lookback = getattr(wrapper, "prediction_days", 3)
    scaler = wrapper.scaler
    s = wrapper.s
    base_model = wrapper.model

    # --- construir historial con padding si hace falta ---
    vals = list(serie_real.values)
    if len(vals) >= lookback:
        historial = vals[:]  # todos los reales
    else:
        # Rellenamos por la izquierda con el primer valor
        pad = [vals[0]] * (lookback - len(vals))
        historial = pad + vals

    preds = []

    for _ in range(horizon):
        # siempre tomamos exactamente 'lookback' pasos
        window_arr = np.array(historial[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scale_window_asinh(window_arr, scaler, s)

        # Keras (TCN / LSTM) vs sklearn (MLP)
        if hasattr(base_model, "input_shape") and base_model.input_shape is not None and len(base_model.input_shape) == 3:
            # modelos tipo Keras secuenciales: (batch, timesteps, features)
            X = window_scaled.reshape(1, lookback, 1)
            y_scaled = base_model.predict(X, verbose=0)[0, 0]
        else:
            # modelos tipo sklearn: (batch, features)
            X = window_scaled.reshape(1, -1)
            y_scaled = base_model.predict(X)[0]

        y_pred = inverse_scale_single(y_scaled, scaler, s)
        historial.append(y_pred)
        preds.append(y_pred)

    idx = pd.date_range(start=pd.to_datetime(start_date), periods=horizon, freq="MS")
    return pd.Series(preds, index=idx)

def plot_predicciones_test(
    codigo: str,
    modelo_nombre: str,
    real_month_test: pd.Series,
    pred_month_test: pd.Series,
    save_path: str = None
):
    """
    Dibuja la gráfica del tramo de TEST: real vs predicción.
    Si save_path se proporciona, guarda la imagen.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(real_month_test.index, real_month_test.values, label="Real", marker="o")
    plt.plot(pred_month_test.index, pred_month_test.values, label=f"Predicción {modelo_nombre}", marker="s")

    plt.title(f"Cuenta {codigo} — Real vs Predicción TEST ({modelo_nombre})")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[OK] Gráfica guardada en {save_path}")

    plt.show()

def append_pred_test_a_csv(
    codigo: str,
    modelo_nombre: str,
    df_mensual: pd.DataFrame,
    results: dict,
    horizonte: int,
    path_salida_csv: str = "prediccion.csv",
):
    """
    Agrega al CSV:
      - una fila REAL (toda la historia mensual de la cuenta)
      - una fila PRED_<modelo> con la predicción FUTURA mensual
        usando results["future_pred"] y results["future_idx"].

    horizonte: máximo de meses futuros a escribir (por ejemplo 57).
    """
    col = str(codigo)

    if col not in df_mensual.columns.astype(str).tolist():
        print(f"[WARN] Cuenta {col} no está en df_mensual. Me la salto en export.")
        return

    # --- 1) REAL completo ---
    serie_real_full = df_mensual[col].astype(float)
    if not isinstance(serie_real_full.index, pd.DatetimeIndex):
        serie_real_full.index = pd.to_datetime(serie_real_full.index)
    serie_real_full = serie_real_full.asfreq("MS").sort_index()

    # --- 2) FUTURO desde results ---
    future_pred = results.get("future_pred", None)
    future_idx = results.get("future_idx", None)

    if future_pred is None or future_idx is None:
        print(f"[WARN] No future_pred/future_idx en results para cuenta {codigo}, modelo {modelo_nombre}")
        return

    future_pred = np.asarray(future_pred).ravel()
    future_idx = pd.DatetimeIndex(future_idx)

    # recortamos al horizonte
    n = min(len(future_pred), len(future_idx), horizonte)
    future_pred = future_pred[:n]
    future_idx = future_idx[:n]

    # --- 3) columnas de meses (ej. '2024-04', etc.) ---
    month_cols = [dt.strftime("%Y-%m") for dt in serie_real_full.index]
    fecha_to_pos = {dt: i for i, dt in enumerate(serie_real_full.index)}

    real_row_vals = serie_real_full.values.tolist()
    pred_vals = [np.nan] * len(serie_real_full)

    for dt_fut, val in zip(future_idx, future_pred):
        if dt_fut in fecha_to_pos:
            pred_vals[fecha_to_pos[dt_fut]] = val

    # --- 4) DataFrame de salida ---
    cols_out = ["Codigo", "modelo", "tipo"] + month_cols
    df_out = pd.DataFrame(columns=cols_out)

    df_out.loc[0] = [codigo, "N/A", "REAL"] + real_row_vals
    df_out.loc[1] = [codigo, modelo_nombre, f"PRED_{modelo_nombre}"] + pred_vals

    header = not os.path.exists(path_salida_csv)
    df_out.to_csv(path_salida_csv, mode="a", index=False, header=header)
    print(f"[OK] Exportadas filas REAL y PRED_{modelo_nombre} para cuenta {codigo} en {path_salida_csv}")




def extraer_pred_test_desde_wrapper(wrapper):
    """
    Toma un wrapper (TCNWrapper / LSTMWrapper / MLPSeriesWrapper) cargado con joblib
    y reconstruye:
      - real_month: serie mensual real en el tramo de test
      - pred_month: serie mensual promedio en el tramo de test
    EXACTAMENTE como en el entrenamiento.
    """
    if wrapper.test_df is None or wrapper.all_preds is None:
        raise ValueError("El wrapper no tiene test_df o all_preds; asegúrate de que se haya entrenado con test.")

    test_df = wrapper.test_df.copy()  # DataFrame con índice datetime y columna 'Adj Close'
    all_preds = wrapper.all_preds     # lista de arrays (n_iters, len(test_df))

    # promedio de todas las corridas
    avg_pred = np.mean(np.stack(all_preds, axis=0), axis=0)

    # serie diaria
    pred_daily = pd.Series(
        np.asarray(avg_pred).ravel(),
        index=test_df.index
    )
    real_daily = test_df["Adj Close"]

    # pasamos ambas a mensual
    pred_month = pred_daily.resample("MS").mean()
    real_month = real_daily.resample("MS").mean()

    return real_month, pred_month


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def predecir(
    horizonte: int,
    codigo: str,
    path_real_csv: str,
    path_model_joblib: str,
    path_salida_csv: str,
    start_forecast: str = "2024-04-01",
):
    """
    Versión que NO inventa un forecast nuevo, sino que:
      - carga el wrapper desde .joblib
      - lee las predicciones de test que ya tiene (2024-04 .. 2025-11 aprox)
      - las pasa a mensual
      - genera CSV con fila REAL y fila PREDICCIÓN
      - grafica real vs predicción.
    El parámetro 'horizonte' aquí se usa sólo para recortar, si quieres,
    pero el tramo base viene del wrapper (test_start/test_end).
    """

    # -------------------------
    # 1) Leer archivo ancho (real completo)
    # -------------------------
    df = pd.read_csv(path_real_csv)

    if "Codigo" not in df.columns:
        raise ValueError("El CSV debe tener columna 'Codigo'.")

    fila = df.loc[df["Codigo"].astype(str) == str(codigo)]
    if fila.empty:
        raise ValueError(f"No existe la cuenta {codigo} en el CSV.")

    fila = fila.iloc[0]

    # columnas de meses (a partir de la 4ª)
    month_cols = df.columns[4:]

    # mapa col -> fecha
    col_to_date = {col: parse_mes_en(col) for col in month_cols}

    # serie completa real (todas las columnas de meses)
    fechas = [col_to_date[c] for c in month_cols]
    valores = pd.to_numeric(fila[month_cols], errors="coerce").values
    serie_all = pd.Series(valores, index=fechas).sort_index()

    # -------------------------
    # 2) Cargar wrapper y extraer predicciones de TEST
    # -------------------------
    wrapper = joblib.load(path_model_joblib)
    real_month_test, pred_month_test = extraer_pred_test_desde_wrapper(wrapper)

    # si quieres recortar al horizonte, lo puedes hacer aquí
    if horizonte is not None and horizonte > 0:
        pred_month_test = pred_month_test.iloc[:horizonte]
        real_month_test = real_month_test.loc[pred_month_test.index]

    print(
        f"Predicción de TEST desde {pred_month_test.index[0].date()} "
        f"hasta {pred_month_test.index[-1].date()} "
        f"({len(pred_month_test)} meses)"
    )

    # -------------------------
    # 3) Construir filas REAL y PREDICCIÓN para el CSV
    # -------------------------
    real_row = []
    pred_row = []

    for col in month_cols:
        dtc = col_to_date[col]

        # real completo: del CSV ancho
        real_row.append(serie_all.get(dtc, np.nan))

        # predicción solo en las fechas de test
        if dtc in pred_month_test.index:
            pred_row.append(pred_month_test[dtc])
        else:
            pred_row.append(np.nan)

    df_out = pd.DataFrame(columns=["Codigo", "tipo"] + list(month_cols))
    df_out.loc[0] = [codigo, "REAL"] + real_row
    df_out.loc[1] = [codigo, "PREDICCION_TEST"] + pred_row

    os.makedirs(os.path.dirname(path_salida_csv), exist_ok=True)
    df_out.to_csv(path_salida_csv, index=False)
    print(f"[OK] CSV guardado en {path_salida_csv}")

    # -------------------------
    # 4) Graficar real completo + predicción de TEST
    # -------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(serie_all.index, serie_all.values, label="Real (completo)", linewidth=1.5)

    # sólo tramo de test
    plt.plot(
        pred_month_test.index,
        pred_month_test.values,
        label="Predicción (TEST wrapper)",
        linewidth=1.8
    )

    # línea vertical en el inicio de test
    test_start = pred_month_test.index[0]
    plt.axvline(test_start, color="gray", linestyle="--", label="Inicio TEST")

    plt.title(f"Cuenta {codigo} — Predicción en TEST (no autoregresiva nueva)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================
if __name__ == "__main__":
    predecir(
        horizonte=15,
        codigo="101101",
        path_real_csv="./dataset/Crediguate.csv",
        path_model_joblib="./instituciones/institucion_35/sucursal_0/101101/TCN_20_101101.joblib",
        path_salida_csv="./pred_101101.csv",
        start_forecast="2024-04-01",
    )
