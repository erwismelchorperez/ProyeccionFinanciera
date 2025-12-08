import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import re
import calendar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from inspect import isfunction
import sys, subprocess
import os

MESES_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}
#VARIABLES
def leer_variables(ruta: str) -> pd.DataFrame:
    """
    Lee un CSV de variables macro con una columna de fecha y varias columnas numéricas.
    Devuelve un DataFrame con índice mensual al inicio de mes (MS).
    """
    df = pd.read_csv(ruta)

    # normalizar nombres
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # detectar columna de fecha
    posibles_fechas = ["fecha", "date", "mes"]
    fecha_col = next((c for c in posibles_fechas if c in df.columns), None)
    if fecha_col is None:
        raise ValueError("No encontré columna de fecha (busqué: fecha, date, mes).")

    # parsear fecha (tus datos son dd/mm/yyyy)
    df[fecha_col] = pd.to_datetime(df[fecha_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[fecha_col]).set_index(fecha_col).sort_index()

    # pasar de "último día del mes" a "primer día del mes"
    # 1) convertir a periodo mensual
    pi = df.index.to_period("M")
    # 2) volver a timestamp al INICIO del periodo
    df.index = pi.to_timestamp(how="start")   # esto te da 2013-01-01, 2013-02-01, ...

    # convertir columnas a numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # aseguramos frecuencia mensual (no rellenamos valores)
    df = df.asfreq("MS")

    return df


def splitsTrainTest_from_df(df: pd.DataFrame,
                            target_col: str,
                            n_lags: int = 3,
                            train_start=None,
                            train_end=None):
    d = df.copy()

    # asegurar datetime
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    # lags del target
    for k in range(1, n_lags + 1):
        d[f"{target_col}_lag{k}"] = d[target_col].shift(k)

    # quitar NaN por lags
    d = d.dropna()

    y = d[target_col].astype(float).values
    X = d.drop(columns=[target_col]).astype(float).values
    idx = d.index

    if train_start is not None and train_end is not None:
        mask_train = (idx >= train_start) & (idx <= train_end)
    else:
        n = len(d)
        n_train = int(n * 0.8)
        mask_train = np.zeros(n, bool)
        mask_train[:n_train] = True

    X_train = X[mask_train]
    y_train = y[mask_train]
    X_test  = X[~mask_train]
    y_test  = y[~mask_train]

    idx_train = idx[mask_train]
    idx_test  = idx[~mask_train]
    return X_train, y_train, X_test, y_test, idx_train, idx_test


def parse_ddmmyyyy(s):
    return pd.to_datetime(s, dayfirst=True, errors='coerce')

def format_mes_yy(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.month.map(MESES_ES) + dt_series.dt.year.mod(100).astype(str).str.zfill(2)

def to_numeric(df, cols, fillna=0.0):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fillna)
    return df

def to_numeric_except(df, exclude=None, fillna=0.0, inplace=True, convert_datetimes=False):
    exclude = set(exclude or [])
    cols = [c for c in df.columns if c not in exclude]

    # opcionalmente evita tocar datetimes
    if not convert_datetimes:
        cols = [c for c in cols if not is_datetime64_any_dtype(df[c])]

    # convierte solo las que NO son numéricas (evita trabajo innecesario)
    cols_to_convert = [c for c in cols if not is_numeric_dtype(df[c])]

    converted = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    if fillna is not None:
        converted = converted.fillna(fillna)

    if inplace:
        df[cols_to_convert] = converted
        return df
    else:
        out = df.copy()
        out[cols_to_convert] = converted
        print(df)
        return out


def drop_cols(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


# --- Parser robusto de etiquetas mensuales ---
_MES_MAP = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,
            "jul":7,"ago":8,"sep":9,"oct":10,"nov":11,"dic":12}

def normalizarEncabezados(df):
    df.columns = (df.columns
                .astype(str)
                .str.replace("\ufeff", "", regex=False)  # quita BOM
                .str.strip()
                .str.lower())

    print("Encabezados normalizados:", list(df.columns))
    return df


def FormatearFecha(df):
    df['fecha'] = df['fecha'].apply(lambda x: formatearColumns(x))
    #print(self.datasetfiltrado)
    return df
def formatearColumns(col):
    return col.replace('31-', '').replace('30-', '').replace('29-', '').replace('28-', '').replace('-', '')
def filtrar(DATASET):
    df=DATASET[DATASET['proyeccion'].str.strip().str.upper()=='SI']
    df= df.drop(columns=['nivel'])
    if 'proyeccion' in df.columns:
        df=df.drop(columns=['proyeccion'])
    if 'balance general' in df.columns:
        df=df.drop(columns=['balance general'])
    df['codigo'] = (
        df['codigo']
            .astype(str)              # fuerza a string
            .str.strip()              # quita espacios
            .str.replace(r'\.0$', '', regex=True)  # elimina sufijo ".0" si quedó por cast desde float
        )
    df = df[
            df['codigo'].ne('') & df['codigo'].ne('nan')
        ].copy()
    df=df.set_index('codigo').T
    df = df.reset_index().rename(columns={'index': 'fecha'})
    df=FormatearFecha(df)
    df=reemplazaGuionPorCERO(df)
    return df
def filtrar(DATASET,flag=True):
    df=DATASET[DATASET['proyeccion'].str.strip().str.upper()=='SI']
    df= df.drop(columns=['nivel'])
    if 'proyeccion' in df.columns:
        df=df.drop(columns=['proyeccion'])
    if 'balance general' in df.columns:
        df=df.drop(columns=['balance general'])
    df['codigo'] = (
        df['codigo']
            .astype(str)              # fuerza a string
            .str.strip()              # quita espacios
            .str.replace(r'\.0$', '', regex=True)  # elimina sufijo ".0" si quedó por cast desde float
        )
    df = df[
            df['codigo'].ne('') & df['codigo'].ne('nan')
        ].copy()
    return df
def reemplazaGuionPorCERO(df):
    df2= df.copy()
    # Reemplazar todos los '-' por 0
    df2= df2.replace('-', 0)
    for col in df2.columns:
        if col not in ["fecha", "codigo"]:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0)
    # FECHA y codigo quedan como están
    df= df2
    return df
def parse_mes_yy(tag: str) -> pd.Timestamp:
    s = str(tag).strip().lower()
    s = s.replace(" ", "").replace("_", "-")
    # ejemplos válidos: ene-13, ene13, ene-2013, ene2013
    m = re.match(r"^([a-záéíóúñ]{3})-?(\d{2}|\d{4})$", s)
    if not m:
        raise ValueError(f"Etiqueta de mes no reconocida: '{tag}'")
    m3, y = m.groups()
    m3 = m3[:3]
    if m3 not in _MES_MAP:
        raise ValueError(f"Mes no reconocido: '{m3}' en '{tag}'")
    year = int(y)
    if year < 100:
        year += 2000
    return pd.Timestamp(year=year, month=_MES_MAP[m3], day=1)
def _asfreq_per_code(g):
    code = g.name  # viene del groupby
    g2 = (g.set_index("Date")[["valor"]]
            .asfreq("MS", fill_value=0.0)
            .rename(columns={"valor": "Adj Close"}))
    g2["Codigo"] = code
    return g2.reset_index()
def read(ruta_csv):
    df = pd.read_csv(ruta_csv, sep=None, engine="python")
    df=filtrar(df)
    #df=normalizarEncabezados(df)
    #df=filtrar(df,True)
    #df=reemplazaGuionPorCERO(df);
    return df
def filtrar(df):
    """
    Lee y preprocesa el dataset completo:
      - Filtra proyeccion=='SI'
      - Elimina columnas: NIVEL, BALANCE GENERAL, proyeccion (si existen)
      - Limpia 'Codigo'
      - Convierte columnas de meses a fechas mensuales y numéricos
      - Devuelve (df_wide, df_long) si keep_format="both"
    """
    # 2) Normalizar encabezados
    df=normalizarEncabezados(df);
    raw=df.copy()
    # 3) Filtro proyeccion==SI (si existe)
    if "proyeccion" in raw.columns:
        raw["proyeccion"] = raw["proyeccion"].astype(str).str.strip().str.upper()
        raw = raw[raw["proyeccion"] == "SI"].copy()

    # 4) Remover columnas no usadas (si existen)
    for col in ("nivel", "balance general", "proyeccion"):
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    # 5) Asegurar 'codigo'
    if "codigo" not in raw.columns:
        raise ValueError("No se encontró la columna 'Codigo' (insensible a mayúsculas).")
    # limpia codigo: string, sin '.0', sin vacíos
    raw["codigo"] = (raw["codigo"]
                     .astype(str)
                     .str.strip()
                     .str.replace(r"\.0$", "", regex=True))
    raw = raw[raw["codigo"].ne("").fillna(False)].copy()

    # 6) Identificar columnas de meses (todas excepto 'codigo')
    month_cols = [c for c in raw.columns if c != "codigo"]

    # 7) Derretir -> formato long
    long = raw.melt(id_vars="codigo", value_vars=month_cols,
                    var_name="mes", value_name="valor")

    # 8) Parsear fecha mensual
    long["Date"] = long["mes"].apply(parse_mes_yy)

    # 9) Limpieza numérica robusta
    #    - deja dígitos / signo / . , ; remueve otros
    #    - "-" o vacíos -> 0
    long["valor"] = (long["valor"].astype(str)
                     .str.replace(r"[^\d\-\.,]", "", regex=True)
                     .str.replace(".", "", regex=False)
                     .str.replace(",", "", regex=False)
                     .str.strip())
    long.loc[long["valor"].eq(""), "valor"] = "0"
    long.loc[long["valor"].str.lower().isin(["-","na","nan","none"]), "valor"] = "0"
    long["valor"] = pd.to_numeric(long["valor"], errors="coerce").fillna(0.0)

    # 10) Asegurar frecuencia mensual continua POR CODIGO y rellenar 0
    #     (para no “perder” meses que no venían en el CSV original)
    long_full = (long[["codigo","Date","valor"]]
                 .sort_values(["codigo","Date"])
                 .groupby("codigo", group_keys=False)
                 .apply(_asfreq_per_code, include_groups=False))

    # 11) Salidas
    df_long = long_full.copy()
    # wide: index=Date, columnas=codigo, valores=Adj Close
    df_wide = (df_long.pivot(index="Date", columns="Codigo", values="Adj Close")
                      .sort_index()
                      .fillna(0.0))
    
    return df_wide;

def aumentarDatoporMes(df,n_dias_agregar:int=10,incluir_original:bool=True,solo_impares:bool=True)-> pd.DataFrame:
    """
    Replica cada fila mensual en varios días del mismo mes.
    - El valor se repite (no interpola).
    - Usa días impares hasta completar n_dias_agregar (o los que existan).
    - Si incluir_original=True, se conserva también el día original (p.ej. 1).

    Devuelve un DataFrame "diario" con el mismo número de columnas que df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index debe ser DatetimeIndex con frecuencia mensual (MS).")

    out_idx, out_rows = [], []

    for ts, row in df.iterrows():
        # último día del mes
        last_day = calendar.monthrange(ts.year, ts.month)[1]

        # candidatos: días impares o todos
        if solo_impares:
            candidatos = [d for d in range(1, last_day + 1, 2)]
        else:
            candidatos = list(range(1, last_day + 1))

        # si quieres preservar el original, quítalo de los candidatos (lo añadimos aparte)
        if incluir_original and ts.day in candidatos:
            candidatos = [d for d in candidatos if d != ts.day]

        # elige tantos días como pidas, respetando el tope del mes
        dias_elegidos = candidatos[:max(0, int(n_dias_agregar))]

        # arma la lista final de días para este mes
        dias_finales = []
        if incluir_original:
            dias_finales.append(ts.day)         # el original
        dias_finales.extend(dias_elegidos)      # los “extras”
        dias_finales = sorted(set(dias_finales))

        # crea las filas replicadas
        for d in dias_finales:
            out_idx.append(ts.replace(day=d))
            out_rows.append(row.values)

    out = pd.DataFrame(out_rows, columns=df.columns,
                       index=pd.DatetimeIndex(out_idx, name="Date")).sort_index()
    print(out.head(24))
    return out

def aumentar_columna_por_mes(df: pd.DataFrame,
                             col: str,
                             n_dias_agregar: int = 15,
                             incluir_original: bool = True,
                             solo_impares: bool = True) -> pd.Series:
    """
    Replica la columna `col` de un DataFrame mensual (index DatetimeIndex MS)
    en varios días del mismo mes (repite el mismo valor).

    Parámetros
    ----------
    df : DataFrame con índice DatetimeIndex (frecuencia mensual, p.ej. 'MS').
    col : nombre de la columna a replicar.
    n_dias_agregar : cuántos días extra por mes (además del original si incluir_original=True).
    incluir_original : si True, incluye también el día original del índice (típicamente 1).
    solo_impares : si True, usa días impares como candidatos (1,3,5,...) hasta completar.

    Retorna
    -------
    Serie diaria (DatetimeIndex) con el valor de `col` replicado en los días elegidos.
    """
    if col not in df.columns:
        raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index debe ser DatetimeIndex.")
    # No forzamos .asfreq('MS') por si ya traes el día 1; si tu índice no es el 1, sigue funcionando.

    out_idx, out_vals = [], []

    for ts, val in df[col].items():
        last_day = calendar.monthrange(ts.year, ts.month)[1]

        # candidatos: días impares o todos
        if solo_impares:
            candidatos = [d for d in range(1, last_day + 1, 2)]
        else:
            candidatos = list(range(1, last_day + 1))

        # si preservas el original, quítalo de candidatos (se agrega aparte)
        if incluir_original and ts.day in candidatos:
            candidatos = [d for d in candidatos if d != ts.day]

        # elige tantos días como pidas
        dias_elegidos = candidatos[:max(0, int(n_dias_agregar))]

        # días finales para este mes
        dias_finales = []
        if incluir_original:
            dias_finales.append(ts.day)
        dias_finales.extend(dias_elegidos)
        dias_finales = sorted(set(dias_finales))

        for d in dias_finales:
            out_idx.append(ts.replace(day=d))
            out_vals.append(val)

    serie = pd.Series(out_vals,
                      index=pd.DatetimeIndex(out_idx, name="Date")).sort_index()
    return serie


def ceros_iniciales_serie(s: pd.Series) -> int:
    """
    Cuenta cuántos valores iniciales son exactamente 0.
    s debe ser una Serie 1D (por ejemplo la columna mensual de una cuenta).
    """
    cnt = 0
    for v in s:
        if float(v) == 0.0:
            cnt += 1
        else:
            break
    return cnt

def aumentar_columna_por_mes_saltando_ceros_iniciales(df: pd.DataFrame,
                                                      col: str,
                                                      n_dias_agregar: int = 15,
                                                      incluir_original: bool = True,
                                                      solo_impares: bool = True) -> pd.Series:
    """
    Igual que la anterior, pero:
    - detecta cuántos meses iniciales son 0
    - NO los aumenta
    - sí aumenta a partir del primer mes distinto de 0
    """
    if col not in df.columns:
        raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index debe ser DatetimeIndex.")

    # serie mensual de esa cuenta
    serie_mensual = df[col]

    # cuántos meses iniciales son 0
    lead_zeros = ceros_iniciales_serie(serie_mensual)

    out_idx, out_vals = [], []

    for i, (ts, val) in enumerate(serie_mensual.items()):
        # mientras estemos en los ceros iniciales → NO aumentar, solo dejar el mes tal cual
        if i < lead_zeros:
            # dejamos solo el día original (para no perder el dato mensual)
            out_idx.append(ts)
            out_vals.append(val)
            continue

        # a partir de aquí, mismo aumento que la función original
        last_day = calendar.monthrange(ts.year, ts.month)[1]

        if solo_impares:
            candidatos = [d for d in range(1, last_day + 1, 2)]
        else:
            candidatos = list(range(1, last_day + 1))

        if incluir_original and ts.day in candidatos:
            candidatos = [d for d in candidatos if d != ts.day]

        dias_elegidos = candidatos[:max(0, int(n_dias_agregar))]

        dias_finales = []
        if incluir_original:
            dias_finales.append(ts.day)
        dias_finales.extend(dias_elegidos)
        dias_finales = sorted(set(dias_finales))

        for d in dias_finales:
            out_idx.append(ts.replace(day=d))
            out_vals.append(val)

    return pd.Series(out_vals,
                     index=pd.DatetimeIndex(out_idx, name="Date")).sort_index()

def all_zero(serie: pd.Series) -> bool:
    """True si todos los valores son 0."""
    v = np.asarray(serie, float)
    return np.all(v == 0)


def splitsTrainTest(serie, n_lags=3, train_ratio=0.8):
    """
    Convierte una serie 1D (mensual) en X, y con lags y la separa en train/test.

    Parameters
    ----------
    serie : array-like o Series
        Serie temporal 1D, ordenada en el tiempo (viejo → nuevo).
    n_lags : int
        Cuántos meses hacia atrás usar como features.
    train_ratio : float
        Proporción para entrenamiento.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    # 1) asegurar que sea array 1D de float
    if isinstance(serie, (pd.Series, pd.DataFrame)):
        valores = pd.Series(serie).astype(float).values
    else:
        valores = np.asarray(serie, float)

    X, y = [], []
    # arrancamos desde n_lags porque necesitamos n_lags valores previos
    for i in range(n_lags, len(valores)):
        X.append(valores[i - n_lags:i])  # [t-3, t-2, t-1]
        y.append(valores[i])             # [t]

    X = np.array(X)   # (N, n_lags)
    y = np.array(y)   # (N,)

    if len(X) == 0:
        raise ValueError("La serie es muy corta para el número de lags indicado.")

    # 2) split temporal
    n = len(X)
    n_train = int(n * train_ratio)
    if n_train == 0 or n_train == n:
        # por si la serie es muy pequeña
        n_train = max(1, n - 1)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test  = X[n_train:]
    y_test  = y[n_train:]

    return X_train, y_train, X_test, y_test



def tiene_negativos(serie: pd.Series) -> bool:
    """True si existe al menos un valor < 0."""
    v = np.asarray(serie, float)
    return np.any(v < 0)

def ceros_iniciales(serie: pd.Series) -> float:
    """
    Cuenta cuántos ceros hay al inicio (corridos) y lo divide entre el largo total.
    Ej: [0,0,0,5,0] -> leading=3 -> 3/5 = 0.6
    """
    v = np.asarray(serie, float)
    n = len(v)
    count = 0
    for val in v:
        if val == 0:
            count += 1
        else:
            break
    return count / n if n else 0.0


def graficar_x_pred_mensual(x, all_preds, test_df, test_start=None,
                            end_limit="2025-06-01", titulo=None):
    """
    x         : DataFrame o Series DIARIA de una cuenta (p.ej. x = serie_101.to_frame('101')).
                Su índice debe ser DatetimeIndex (días). El real mensual se obtiene promediando por mes.
    all_preds : lista de arrays predichos (cada uno con shape (N_test,)), que ya promediarás.
    test_df   : DataFrame/Series con índice DIARIO para el tramo de test (solo usamos el índice).
    test_start: datetime opcional para trazar una línea vertical de inicio de test.
    end_limit : fecha tope (string o Timestamp) para recortar ambas curvas.
    titulo    : título del gráfico.

    Devuelve: dict con pred_month, real_month y rmse_month.
    """
    # -------- Real (desde x diaria) -> mensual --------
    if isinstance(x, pd.Series):
        real_daily = x.copy()
        value_name = x.name if x.name else "valor"
    else:
        # DataFrame: tomamos la única columna
        if x.shape[1] != 1:
            raise ValueError("x debe tener una sola columna (una cuenta).")
        value_name = x.columns[0]
        real_daily = x.iloc[:, 0].copy()

    if not isinstance(real_daily.index, pd.DatetimeIndex):
        real_daily.index = pd.to_datetime(real_daily.index)
    real_daily = real_daily.sort_index()

    if end_limit is not None:
        real_daily = real_daily.loc[:pd.to_datetime(end_limit)]

    # Real mensual (promedio)
    real_month = real_daily.resample("MS").mean()

    # -------- Predicción (desde all_preds + test_df.index) --------
    if not hasattr(test_df, "index"):
        raise ValueError("test_df debe tener un índice (DatetimeIndex) del tramo de test diario.")
    pred_avg = np.mean(np.stack(all_preds, axis=0), axis=0)          # (N_test,)
    pred_daily = pd.Series(np.asarray(pred_avg).ravel(), index=test_df.index)

    if not isinstance(pred_daily.index, pd.DatetimeIndex):
        pred_daily.index = pd.to_datetime(pred_daily.index)
    pred_daily = pred_daily.sort_index()

    if end_limit is not None:
        pred_daily = pred_daily.loc[:pd.to_datetime(end_limit)]

    # Pred mensual (promedio)
    pred_month = pred_daily.resample("MS").mean()

    # -------- Alineación para métrica --------
    common_idx = real_month.index.intersection(pred_month.index)
    real_m = real_month.loc[common_idx]
    pred_m = pred_month.loc[common_idx]

    if len(common_idx) == 0:
        rmse = np.nan
        print("No hay meses comunes para calcular RMSE.")
    else:
        mse = mean_squared_error(real_m.values, pred_m.values)
        rmse = float(np.sqrt(mse))
        print(f"RMSE mensual: {rmse:.4f}")

    # -------- Gráfico principal --------
    plt.figure(figsize=(12,5))
    plt.plot(real_month.index, real_month.values,
             label=f"Real (mensual) — {value_name}", color="black", linewidth=1.8)
    plt.plot(pred_month.index, pred_month.values,
             label="Predicción (promedio mensual)", linewidth=2)

    if test_start is not None:
        ts = pd.to_datetime(test_start)
        ts_m = pd.Timestamp(ts.year, ts.month, 1)  # evita el error de 'MS' como freq de Period
        plt.axvline(ts_m, linestyle="--", alpha=0.7, label="Inicio test")

    plt.title(titulo or f"Cuenta {value_name}: Real mensual vs Pred mensual")
    plt.xlabel("Fecha"); plt.ylabel("Valor")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # -------- Gráfico zoom en tramo de test (opcional) --------
    # Tomamos solo meses presentes en el tramo test (según pred_month)
    if len(pred_month) > 0:
        zoom_idx = real_month.index.intersection(pred_month.index)
        if len(zoom_idx) > 0:
            plt.figure(figsize=(12,5))
            plt.plot(real_month.loc[zoom_idx].index, real_month.loc[zoom_idx].values,
                     label="Real (mensual) — test", color="black", linewidth=1.8)
            plt.plot(pred_month.loc[zoom_idx].index, pred_month.loc[zoom_idx].values,
                     label="Pred (promedio mensual) — test", linewidth=2)
            if test_start is not None:
                ts = pd.to_datetime(test_start)
                ts_m = pd.Timestamp(ts.year, ts.month, 1)
                plt.axvline(ts_m, linestyle="--", alpha=0.7, label="Inicio test")
            plt.title((titulo or f"Cuenta {value_name}") + " — Zoom test")
            plt.xlabel("Fecha"); plt.ylabel("Valor")
            plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    return {"pred_month": pred_month, "real_month": real_month, "rmse_month": rmse}

def choose_models(
    base_models,
    *,
    all_zero_val: bool,
    has_neg: bool,
    lead_zero_ratio: float,
    zero_ratio: float,
    lead_thresh: float = 0.60,
    high_zero_thresh: float = 0.50,
    tcn_max_zero_ratio: float = 0.95 #TOLERANCIA A LOS 0's
):
    """
    Devuelve un dict de modelos a entrenar según propiedades de la serie.

    base_models: dict con TODOS tus modelos posibles, por ejemplo:
        {
            "ZeroInflatedPoisson": ...,
            "Lightgbm": ...,
            "TwoPart": ...,
            "Lasso": ...,
            ...
            "TCN": TCNWrapper()   # <-- NN
        }
    """
    print(base_models)
    # 1) caso extremo: todo es 0
    print(all_zero_val)
    print(has_neg)
    print(zero_ratio)
    print(lead_zero_ratio)

    if all_zero_val:
        from src.new_models.AlwaysZero import AlwaysZeroWrapper  # ajusta el import
        return {"AlwaysZero": AlwaysZeroWrapper()}

    modelos = {}

    # 2) si hay negativos: descarta los que no toleran (ZIP, Tweedie)
    if has_neg:
        # TCN sí lo puedes usar aquí porque tú ya haces asinh + MinMax antes
        for name in (
            "TwoPart",
            "Lasso", "LassoPSO",
            "Linear", "LinearPSO",
            "Ridge", "RidgePSO",
            "LSTM",
            "TCN",
            "MLP"
        ):
            if name in base_models:
                modelos[name] = base_models[name]
        return modelos

    # 3) no hay negativos
    if zero_ratio >= high_zero_thresh:
        # Mucho cero → modelos "zero-aware"
        if "TwoPart" in base_models:
            modelos["TwoPart"] = base_models["TwoPart"]
        if "ZeroInflatedPoisson" in base_models:
            modelos["ZeroInflatedPoisson"] = base_models["ZeroInflatedPoisson"]
        if "Ridge" in base_models:
            modelos["Ridge"] = base_models["Ridge"]

        # TCN solo si no es EXCESIVO el cero global
        if zero_ratio <= tcn_max_zero_ratio and "TCN" in base_models:
            modelos["TCN"] = base_models["TCN"]
        # lstm solo si no es EXCESIVO el cero global
        if zero_ratio <= tcn_max_zero_ratio and "LSTM" in base_models:
            modelos["LSTM"] = base_models["LSTM"]
        if zero_ratio <= tcn_max_zero_ratio and "MLP" in base_models:
            modelos["MLP"] = base_models["MLP"]
    else:
        # pocos ceros → casi todo
        modelos = {k: base_models[k] for k in base_models.keys()}

    # 4) si los ceros están muy al inicio, podemos quitar LightGBM y también TCN
    if lead_zero_ratio >= lead_thresh:
        if "Lightgbm" in modelos:
            modelos.pop("Lightgbm")
        # si hay muchos ceros al inicio, el TCN ve puro 0 en las primeras ventanas
        #if "TCN" in modelos and zero_ratio > 0.0:
        #    modelos.pop("TCN")
        #if "TCN" in modelos and zero_ratio > 0.0:
        #    modelos.pop("TCN")

    return modelos


import numpy as np
import pandas as pd

def normalizar_resultado_para_export(results: dict,
                                     df_real: pd.DataFrame,
                                     meses_objetivo: int = 15):
    """
    Normaliza el dict que devolvió un modelo para que SIEMPRE tenga:
      - y_true: np.array de largo N
      - y_pred: np.array de largo N
      - idx: índice de fechas (DatetimeIndex) de largo N
    y N será como máximo `meses_objetivo` y como máximo lo que haya en el real.

    df_real: DataFrame mensual de la cuenta (1 sola columna) con TODO el histórico.
    """
    # 1) sacar arrays que vengan
    y_true = np.asarray(results.get("y_true", []), float).ravel()
    y_pred = np.asarray(results.get("y_pred", []), float).ravel()

    # si no hay nada, devolvemos vacío pero marcado
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "idx": pd.DatetimeIndex([]),
        }

    # 2) igualar longitudes por seguridad
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # 3) queremos como mucho `meses_objetivo` (p.ej. 15 meses: 2024-02 → 2025-04)
    n = min(n, meses_objetivo)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # 4) construir índice de fechas desde el real
    # df_real debe tener índice datetime mensual
    real_series = df_real.iloc[:, 0]
    if not isinstance(real_series.index, pd.DatetimeIndex):
        real_series.index = pd.to_datetime(real_series.index)
    real_series = real_series.asfreq("MS")

    # tomamos los ÚLTIMOS n meses del real
    if len(real_series) >= n:
        idx = real_series.index[-n:]
    else:
        # fallback: generar fechas sintéticas
        idx = pd.date_range(start="2000-01-01", periods=n, freq="MS")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "idx": idx,
    }


def splitsTrainTest_from_series(serie_mensual: pd.Series,
                                n_lags: int = 3,
                                train_start: pd.Timestamp = "2013-01-01",
                                train_end: pd.Timestamp = "2024-01-01"):
    """
    serie_mensual: pd.Series con índice datetime mensual (MS) y un valor por mes
    devuelve: X_train, y_train, X_test, y_test, train_series, test_series
    """
    s = serie_mensual.sort_index()

    # recorte por fechas
    train_s = s.loc[(s.index >= train_start) & (s.index <= train_end)].copy()
    test_s  = s.loc[s.index > train_end].copy()

    # función interna de lags sobre una serie
    def _make_lags(series, n_lags):
        vals = series.values.astype(float)
        X, y = [], []
        for i in range(n_lags, len(vals)):
            X.append(vals[i-n_lags:i])
            y.append(vals[i])
        X = np.array(X, float)
        y = np.array(y, float)
        # alinear índices
        idx = series.index[n_lags:]
        return X, y, idx

    X_train, y_train, train_idx = _make_lags(train_s, n_lags)
    X_test,  y_test,  test_idx  = _make_lags(pd.concat([train_s.tail(n_lags), test_s]), n_lags)

    return X_train, y_train, X_test, y_test, train_s, test_s



def plot_serie_completa_con_model_scores(
    real_full: pd.Series | pd.DataFrame,
    model_scores: dict,
    test_index: pd.DatetimeIndex | None = None,
    titulo: str = "Serie completa + modelos",
    plots_dir: str | None = None,
    filename: str | None = None,
):
    """
    Dibuja real vs. predicciones de todos los modelos y opcionalmente guarda el PNG.

    - plots_dir: carpeta donde guardar el plot (si es None, solo muestra).
    - filename: nombre del archivo, ej. "cuenta_101.png".
      Si no lo pasas pero sí hay título, lo generamos a partir del título.
    """

    if not model_scores:
        print("model_scores está vacío, no hay nada que graficar.")
        return

    # --- 1) normalizar real_full a Serie ---
    if isinstance(real_full, pd.DataFrame):
        if real_full.shape[1] != 1:
            raise ValueError("real_full tiene varias columnas; pásame solo la columna de esa cuenta.")
        real_series = real_full.iloc[:, 0]
    else:
        real_series = real_full

    if not isinstance(real_series.index, pd.DatetimeIndex):
        real_series.index = pd.to_datetime(real_series.index)

    # --- 2) plot de toda la serie ---
    plt.figure(figsize=(13, 5))
    plt.plot(
        real_series.index,
        real_series.values,
        label="Real (completa)",
        color="black",
        linewidth=1.6
    )

    # paleta
    palette = [
        "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:blue", "tab:brown", "tab:pink", "tab:gray",
    ]
    color_iter = iter(palette)

    # --- 3) cada modelo en su tramo ---
    for i, (name, res) in enumerate(model_scores.items()):
        if "y_pred" not in res:
            continue
        y_pred = np.asarray(res["y_pred"]).ravel()

        # usar índice de test si lo tienes
        if test_index is not None:
            m = min(len(test_index), len(y_pred))
            idx = test_index[:m]
            vals = y_pred[:m]
        else:
            # si no, lo pegamos al final del real
            m = min(len(real_series), len(y_pred))
            idx = real_series.index[-m:]
            vals = y_pred[-m:]

        plt.plot(
            idx,
            vals,
            label=name,
            linewidth=1.8,
            color=next(color_iter, None)
        )

    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --- 4) guardar si se pidió ---
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        if filename is None:
            # algo simple a partir del título
            safe_title = titulo.lower().replace(" ", "_").replace("—", "_").replace("–", "_")
            filename = f"{safe_title}.png"
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] gráfica guardada en: {save_path}")

    # mostrar igual
    #plt.show()


def plot_cuenta_completa_vs__model_scores(
    real_full: pd.Series | pd.DataFrame,
    model_scores: dict,
    test_index: pd.DatetimeIndex | None = None,
    titulo: str = "Serie completa + modelos"
):
    if not model_scores:
        print("model_scores está vacío, no hay nada que graficar.")
        return

    # --- real a serie ---
    if isinstance(real_full, pd.DataFrame):
        if real_full.shape[1] != 1:
            raise ValueError("real_full tiene varias columnas; pásame solo la columna de esa cuenta.")
        real_series = real_full.iloc[:, 0]
    else:
        real_series = real_full

    if not isinstance(real_series.index, pd.DatetimeIndex):
        real_series.index = pd.to_datetime(real_series.index)

    plt.figure(figsize=(13, 5))
    plt.plot(real_series.index, real_series.values,
             label="Real (completa)", color="black", linewidth=1.6)

    palette = [
        "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:blue", "tab:brown", "tab:pink", "tab:gray",
    ]
    color_iter = iter(palette)

    for name, res in model_scores.items():
        if "y_pred" not in res:
            continue

        y_pred = np.asarray(res["y_pred"]).ravel()

        # 1) si el modelo ya trae su índice, úsalo
        if "idx" in res and res["idx"] is not None:
            idx = pd.to_datetime(res["idx"])
            m = min(len(idx), len(y_pred))
            idx = idx[:m]
            vals = y_pred[:m]
        # 2) si el usuario pasó test_index al plot, úsalo
        elif test_index is not None:
            m = min(len(test_index), len(y_pred))
            idx = test_index[:m]
            vals = y_pred[:m]
        # 3) fallback: pegar al final del real
        else:
            m = min(len(real_series), len(y_pred))
            idx = real_series.index[-m:]
            vals = y_pred[-m:]

        plt.plot(
            idx,
            vals,
            label=name,
            linewidth=1.8,
            color=next(color_iter, None)
        )

    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_resultados_modelos(res_dict, titulo="Comparación modelos",plots_dir: str | None = None,filename: str | None = None):
    """
    res_dict: diccionario como el que mostraste:
      {
        "Linear": {...},
        "TCN": {...}
      }
    Cada subdict debe tener al menos:
      - 'y_true'
      - 'y_pred'
    """
    # 1) juntar lo que haya
    modelos = list(res_dict.keys())
    series = {}

    for name in modelos:
        y_true = np.asarray(res_dict[name]["y_true"]).ravel()
        y_pred = np.asarray(res_dict[name]["y_pred"]).ravel()
        series[name] = (y_true, y_pred)

    # 2) elegir una serie "real" de referencia
    #    tomamos la primera que tenga y_true
    ref_name = modelos[0]
    y_true_ref = series[ref_name][0]

    # 3) para evitar problemas de longitud, buscamos el mínimo largo
    min_len = min(len(s[0]) for s in series.values())
    min_len = min(min_len, min(len(s[1]) for s in series.values()))

    # 4) recortamos todo
    y_real = y_true_ref[:min_len]

    plt.figure(figsize=(12, 5))
    plt.plot(range(min_len), y_real, label="Real", color="black", linewidth=2)

    # 5) plot de cada modelo
    colores = {
        "Linear": "tab:orange",
        "TCN": "tab:green",
        "LSTM": "tab:red",
    }
    for name in modelos:
        y_pred = series[name][1][:min_len]
        plt.plot(range(min_len), y_pred,
                 label=name, linewidth=1.8,
                 color=colores.get(name, None))

    plt.title(titulo)
    plt.xlabel("Tiempo (índice)")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # --- 4) guardar si se pidió ---
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        if filename is None:
            # algo simple a partir del título
            safe_title = titulo.lower().replace(" ", "_").replace("—", "_").replace("–", "_")
            filename = f"{safe_title}.png"
        save_path = os.path.join(plots_dir, filename)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] gráfica guardada en: {save_path}")
    #plt.show()

def escalar(df_3por_mes):
    # 1) Preprocesado: SOLO asinh (acepta negativos)
    x = df_3por_mes['Adj Close'].astype(float).values.reshape(-1,1)
    # --- MinMaxScaler: fit SOLO con TRAIN, transform en TEST ---
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_ratio = 0.8
    n = len(x); n_train = int(n*train_ratio)

    s = np.nanmedian(np.abs(x[:n_train])) or 1.0   # escala robusta con TRAIN
    x_asinh = np.arcsinh(x / s)
    scaled_data = np.empty_like(x_asinh)
    scaled_data[:n_train] = scaler.fit_transform(x_asinh[:n_train])
    scaled_data[n_train:] = scaler.transform(x_asinh[n_train:])

    # Split en asinh
    train_data = x_asinh[:n_train]
    test_data  = x_asinh[n_train:]
    
    return train_data,test_data,scaler

def escalar_asinh(df_3por_mes):
    # --- Serie base ---
    x = df_3por_mes['Adj Close'].astype(float).values.reshape(-1, 1)

    # --- Split índices primero (80/20) ---
    train_ratio = 0.8
    n = len(x)
    n_train = int(n * train_ratio)

    # --- Transformación asinh (acepta negativos) ---
    # Escala de referencia robusta calculada SOLO con TRAIN
    s = np.nanmedian(np.abs(x[:n_train])) or 1.0
    x_asinh = np.arcsinh(x / s)

    # --- MinMaxScaler: fit SOLO con TRAIN, transform en TEST ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = np.empty_like(x_asinh)
    scaled_data[:n_train] = scaler.fit_transform(x_asinh[:n_train])
    scaled_data[n_train:] = scaler.transform(x_asinh[n_train:])
    asinh_train_min = float(x_asinh[:n_train].min())
    asinh_train_max = float(x_asinh[:n_train].max())

    # Ahora arma train/test en escalado (sin fuga)
    train_data = scaled_data[:n_train]
    test_data  = scaled_data[n_train:]
    return train_data,test_data,scaler

def escalar_asinh_vector(x, train_ratio=0.8):
    """
    x: np.array shape (T,1) con valores (puede tener negativos).
    Devuelve: train_data, test_data, scaler_minmax, s, n_train
    - s: escala robusta usada en asinh (guárdala para invertir luego)
    - n_train: índice de corte (útil si quieres reconstruir)
    """
    x = np.asarray(x, float).reshape(-1, 1)
    n = len(x)
    n_train = int(n * train_ratio) if n > 0 else 0

    # Escala robusta SOLO con train (evita fuga de información)
    s = np.nanmedian(np.abs(x[:n_train])) if n_train > 0 else 1.0
    if not np.isfinite(s) or s == 0:
        s = 1.0

    x_asinh = np.arcsinh(x / s)

    # MinMax SOLO con train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = np.empty_like(x_asinh)
    if n_train > 0:
        scaled[:n_train] = scaler.fit_transform(x_asinh[:n_train])
        if n_train < n:
            scaled[n_train:] = scaler.transform(x_asinh[n_train:])
    else:
        # todo es test -> fit a valores actuales para evitar error
        scaled[:] = scaler.fit_transform(x_asinh)

    train_data = scaled[:n_train]
    test_data  = scaled[n_train:]
    return train_data, test_data, scaler, float(s), n_train

def make_windows(arr, lookback=120, horizon=1):  # 24
    import numpy as np
    a = np.asarray(arr).reshape(-1)  # asegura vector 1D
    X, y = [], []
    for i in range(lookback, len(a) - horizon + 1):
        X.append(a[i - lookback:i].reshape(lookback, 1))  # (lookback, 1)
        y.append([a[i + horizon - 1]])                    # (1,)
    return np.array(X), np.array(y)

def makewindows(train_data, test_data):
    lookback = 120  # 60
    horizon  = 1
    prediction_days = lookback

    X_train, y_train = make_windows(train_data, lookback, horizon)
    X_test,  y_test  = make_windows(test_data,  lookback, horizon)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)  # (N, 60, 1) y (N, 1)
    return X_train, y_train, X_test, y_test, prediction_days


def alinear_por_indice(test_df, pred_array):
    """
    test_df: DataFrame con la columna 'Adj Close' y el índice de las fechas reales
    pred_array: np.array de la misma longitud o mayor (lo que sale de TCN)
    devuelve (y_true_align, y_pred_align) con la misma longitud
    """
    y_true = test_df['Adj Close']
    y_pred = pd.Series(pred_array.ravel(), index=y_true.index)

    # si por alguna razón y_pred tiene más índices, recortamos al común
    idx_comun = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[idx_comun]
    y_pred = y_pred.loc[idx_comun]
    return y_true.values, y_pred.values


def alinear_por_indice(test_df, pred_array):
    """
    test_df: DataFrame con la columna 'Adj Close' y el índice de las fechas reales
    pred_array: np.array de la misma longitud o mayor (lo que sale de TCN)
    devuelve (y_true_align, y_pred_align) con la misma longitud
    """
    y_true = test_df['Adj Close']
    y_pred = pd.Series(pred_array.ravel(), index=y_true.index)

    # si por alguna razón y_pred tiene más índices, recortamos al común
    idx_comun = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[idx_comun]
    y_pred = y_pred.loc[idx_comun]
    return y_true.values, y_pred.values


def plot_modelos_alineados(test_df, resultados_dict, titulo="Comparación"):
    # test_df: DF con Adj Close y el índice de la verdad
    real = test_df['Adj Close']

    plt.figure(figsize=(12,5))
    plt.plot(real.index, real.values, label='Real', color='black', linewidth=2)

    for name, res in resultados_dict.items():
        y_pred = res['y_pred']
        # alineamos
        y_true_al, y_pred_al = alinear_por_indice(test_df, y_pred)
        # para graficar usamos el índice alineado
        idx_comun = test_df.index.intersection(test_df.index)  # solo para claridad
        plt.plot(test_df.index[:len(y_pred_al)], y_pred_al, label=name)

    plt.title(titulo)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# ================== ACTUALIZADOS ==================
def putTest(df_wide, col, prediction_days, scaler, s,
            test_start=dt.datetime(2024,1,1), test_end=dt.datetime(2025,6,1)):
    """
    df_wide: DataFrame con índice datetime y columnas=códigos (df_num)
    col: código/cuenta a usar, e.g. '101' o 101 (se convierte a str)
    scaler: MinMaxScaler YA entrenado sobre asinh(train)
    s: escala robusta usada en el entrenamiento (la misma con la que hiciste asinh)
    """
    col = str(col)
    if col not in df_wide.columns.astype(str).tolist():
        raise ValueError(f"La columna '{col}' no está en df_wide.columns")

    # series crudas
    serie_full = df_wide[col].astype(float).to_frame(name='Adj Close')

    # ventana para test (reales) y contexto para armar inputs
    test_df = serie_full.loc[test_start:test_end].copy()
    if test_df.empty:
        raise ValueError("El rango de test no tiene datos en df_wide.")

    # Toma 'prediction_days' pasos ANTES del inicio de test + todo test
    # para construir model_inputs del tamaño correcto
    ventana = serie_full.iloc[-(len(test_df) + prediction_days):].values.reshape(-1,1)

    # APLICAR MISMA NORMALIZACIÓN QUE EN TRAIN: asinh -> MinMax
    model_inputs = forward_scale(ventana, s, scaler)          # (prediction_days + len(test_df), 1)
    return model_inputs, test_df, test_start, test_end



def putTest_cuenta(x, prediction_days, scaler, s,
                      test_start=dt.datetime(2024,1,1),
                      test_end=dt.datetime(2025,6,1)):
    """
    x: Serie o DataFrame (UNA sola columna) con índice datetime diario (serie aumentada).
       Si es DataFrame con 1 columna, se usa esa columna.
    prediction_days: lookback que necesita tu modelo para armar x_test (contexto previo).
    scaler: MinMaxScaler YA entrenado sobre asinh(train)
    s: escala robusta usada en el entrenamiento (la misma con la que hiciste asinh)
    test_start, test_end: ventana de test real (diaria) sobre x
    """
    # --- 1) Normaliza a Serie ---
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("x debe ser Serie o DataFrame de UNA sola columna.")
        series = x.iloc[:, 0].astype(float).copy()
    else:
        series = x.astype(float).copy()

    # --- 2) Índice datetime y orden ---
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    # --- 3) Ventana de test diaria ---
    test_mask = (series.index >= pd.to_datetime(test_start)) & (series.index <= pd.to_datetime(test_end))
    test_df = series.loc[test_mask]
    if test_df.empty:
        raise ValueError("El rango de test no tiene datos en la serie aumentada.")

    # --- 4) Construir ventana: prediction_days previos a test_start + todo el test ---
    idx = series.index
    values = series.values.reshape(-1, 1)

    # posición donde inicia el test
    pos_start = idx.searchsorted(pd.to_datetime(test_start), side="left")
    pos_end   = idx.searchsorted(pd.to_datetime(test_end),   side="right")  # exclusivo
    test_len  = pos_end - pos_start

    if test_len <= 0:
        raise ValueError("No pude localizar índices de test coherentes en la serie.")

    # Queremos: [pos_start - prediction_days, pos_end)
    if pos_start >= prediction_days:
        ventana = values[pos_start - prediction_days : pos_end]
    else:
        # Falta historia al principio -> pad con el primer valor
        pad_needed = prediction_days - pos_start
        first_val = values[0:1].copy()
        prepad = np.repeat(first_val, pad_needed, axis=0)
        ventana = np.vstack([prepad, values[:pos_end]])

    # --- 5) Escalado forward: asinh con 's' + MinMax (mismo scaler de train) ---
    # forward_scale esperado: arr -> asinh(arr/s) -> scaler.transform(...)
    # si no lo tienes, puedes inlinearlo:
    asinh = np.arcsinh(ventana / float(s))
    model_inputs = scaler.transform(asinh)

    # --- 6) test_df como DataFrame con columna 'Adj Close' ---
    test_df = pd.DataFrame({"Adj Close": test_df.values}, index=test_df.index)

    return model_inputs, test_df, pd.to_datetime(test_start), pd.to_datetime(test_end)


def forward_scale(x_raw, s, scaler):
    # x_raw: (N,1) valores crudos
    x_asinh = np.arcsinh(x_raw / s)
    return scaler.transform(x_asinh)

def inverse_scale(y_scaled, s, scaler):
    y_asinh = scaler.inverse_transform(y_scaled)
    return np.sinh(y_asinh) * s

def predictionTest(prediction_days, model_inputs, model, scaler, s):
    """
    Devuelve X para test y las predicciones ya invertidas a escala original.
    """
    x1_test = []
    for i in range(prediction_days, len(model_inputs)):
        x1_test.append(model_inputs[i - prediction_days:i, 0])
    x1_test = np.array(x1_test).reshape(-1, prediction_days, 1)

    yhat_scaled = model.predict(x1_test, verbose=0)          # (N_test, 1) en espacio MinMax(asinh)
    yhat = inverse_scale(yhat_scaled, s, scaler).reshape(-1) # de vuelta a escala cruda
    return x1_test, yhat


def plot_cuenta_mensual_vs_diario(df_mensual: pd.DataFrame,df_diario: pd.DataFrame,cuenta: str | int,titulo: str | None = None):
    """
    df_mensual: DataFrame ancho (index mensual, columnas=códigos)
    df_diario:  DataFrame ancho (index con múltiples días por mes, mismas columnas)
    cuenta:     código de la cuenta a graficar (str o int)
    """
    c = str(cuenta)
    if c not in df_mensual.columns:
        raise KeyError(f"La cuenta {c} no está en df_mensual.columns")

    if c not in df_diario.columns:
        raise KeyError(f"La cuenta {c} no está en df_diario.columns")

    # series
    s_m = df_mensual[c].astype(float)
    s_d = df_diario[c].astype(float)

    plt.figure(figsize=(12, 5))
    # Mensual en negro (pocos puntos)
    plt.plot(s_m.index, s_m.values, color="black", linewidth=2, marker="o",
             label=f"Mensual ({c})")
    # Diario replicado (muchos puntos)
    plt.plot(s_d.index, s_d.values, linewidth=1, alpha=0.7,
             label=f"Aumentado diario ({c})")

    if titulo is None:
        titulo = f"Cuenta {c} — Mensual vs Aumentado"
    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()