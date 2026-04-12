#import os
import time
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

# === TUS UTILIDADES / ESTRUCTURA DE PROYECTO ===
from src.data_loader_ import Loader
from src.storage import crear_carpeta_cuenta,crear_carpeta_institucion

# ===========================
#   P L O T   M E N S U A L
# ===========================
# ===========================
def plot_monthly(
    real: pd.Series,
    pred: pd.Series,
    train_end: pd.Timestamp,
    title: str,
    save_path: Optional[str] = None,
):
    # --- REAL ---
    if not isinstance(real.index, pd.DatetimeIndex):
        real.index = pd.to_datetime(real.index, errors="coerce")
    real = real.dropna()
    real = real.sort_index()

    # ✅ quitar duplicados (si hay varias filas mismo mes)
    # opción 1: último valor del mes (saldos)
    real = real.groupby(real.index.to_period("M")).last()
    real.index = real.index.to_timestamp(how="start")

    # asegurar frecuencia mensual (sin reindex con duplicados)
    real = real.asfreq("MS")

    # --- PRED ---
    if not isinstance(pred.index, pd.DatetimeIndex):
        pred.index = pd.to_datetime(pred.index, errors="coerce")
    pred = pred.dropna()
    pred = pred.sort_index()

    # ✅ quitar duplicados en pred también
    pred = pred.groupby(pred.index.to_period("M")).last()
    pred.index = pred.index.to_timestamp(how="start")
    pred = pred.asfreq("MS")

    # --- PLOT ---
    plt.figure(figsize=(13, 5))
    plt.plot(real.index, real.values, label="Real", linewidth=1.8)
    plt.plot(pred.index, pred.values, label="Pred", linewidth=1.8)
    plt.axvline(train_end, color="gray", linestyle="--", linewidth=1.0, label="Train end")

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] plot guardado: {save_path}")
    plt.close()



# =========================================
#  C O N S T R U C C I Ó N   D E   F E A T S
#  (MIMO mensual)
# =========================================
def _ensure_ms_index(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[~s.index.isna()].sort_index()

    #  colapsar duplicados a mensual (elige last o sum según tu caso)
    if isinstance(s, pd.Series):
        s = s.groupby(s.index.to_period("M")).last()
        s.index = s.index.to_timestamp(how="start")
        return s.asfreq("MS").sort_index()
    else:
        s = s.groupby(s.index.to_period("M")).last()
        s.index = s.index.to_timestamp(how="start")
        return s.asfreq("MS").sort_index()


def _make_supervised_mimo(
    y_m: pd.Series,
    X_m: Optional[pd.DataFrame],
    lookback: int,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    y_m = _ensure_ms_index(y_m)
    if X_m is not None:
        X_m = _ensure_ms_index(X_m).reindex(y_m.index)

    values = y_m.values.astype(float)
    n = len(values)
    rows = []
    targets = []
    idx_rows = []

    for end_idx in range(lookback, n - horizon + 1):
        hist_slice = values[end_idx - lookback : end_idx]

        if X_m is not None:
            x_exog = X_m.iloc[end_idx - 1].values.astype(float)
            row = np.concatenate([hist_slice, x_exog])
        else:
            row = hist_slice

        y_future = values[end_idx : end_idx + horizon]

        rows.append(row)
        targets.append(y_future)
        idx_rows.append(y_m.index[end_idx])

    X_df = pd.DataFrame(rows)
    Y_df = pd.DataFrame(targets)

    # ✅ Compatible con pandas viejo: NO usar "MS"
    idx_start_rows = pd.DatetimeIndex(pd.to_datetime(idx_rows)).to_period("M").to_timestamp(how="start")

    return X_df, Y_df, idx_start_rows



def _split_train_test_by_date(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    idx_start_rows: pd.DatetimeIndex,
    train_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """
    TRAIN: filas cuyo primer mes pronosticado <= train_end
    TEST: resto
    """
    mask_train = idx_start_rows <= pd.Timestamp(train_end)
    X_tr = X_df.loc[mask_train].copy()
    Y_tr = Y_df.loc[mask_train].copy()
    X_te = X_df.loc[~mask_train].copy()
    Y_te = Y_df.loc[~mask_train].copy()
    idx_te = idx_start_rows[~mask_train]
    return X_tr, Y_tr, X_te, Y_te, idx_te


# ================================
#   M O D E L O   (Ridge / MLP)
# ================================
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def _build_model(model_type: str, random_state: int = 42):
    if model_type == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=random_state))
        ])
    elif model_type == "mlp":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=800,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=20
            ))
        ])
    else:
        raise ValueError("model_type debe ser 'ridge' o 'mlp'")
    return model


# ==========================================
#  T R A I N   /   E V A L   (M I M O)
# ==========================================
def train_mimo_monthly(
    y_monthly: pd.Series,
    X_monthly: Optional[pd.DataFrame],
    train_end: pd.Timestamp,
    lookback: int,
    horizon: int,
    model_type: str = "ridge",
) -> Dict:
    y_monthly = _ensure_ms_index(y_monthly)
    if X_monthly is not None:
        X_monthly = _ensure_ms_index(X_monthly).reindex(y_monthly.index)

    X_df, Y_df, idx_start_rows = _make_supervised_mimo(
        y_m=y_monthly,
        X_m=X_monthly,
        lookback=lookback,
        horizon=horizon,
    )

    X_tr, Y_tr, X_te, Y_te, idx_te = _split_train_test_by_date(
        X_df, Y_df, idx_start_rows, train_end
    )

    if len(X_tr) == 0:
        raise ValueError("No hay filas de entrenamiento con el corte indicado.")

    model = _build_model(model_type=model_type)
    t0 = time.perf_counter()
    model.fit(X_tr.values, Y_tr.values)
    train_time = time.perf_counter() - t0

    if len(X_te) > 0:
        Y_hat = model.predict(X_te.values)  # [n_blocks, horizon]

        idx_all, pred_all = [], []
        for i, t0_month in enumerate(idx_te):
            months = pd.date_range(t0_month, periods=horizon, freq="MS")
            idx_all.append(months)
            pred_all.append(Y_hat[i])

        idx_concat = pd.DatetimeIndex(np.concatenate([idx.values for idx in idx_all]))
        ypred_concat = np.concatenate(pred_all).astype(float)

        real_concat = y_monthly.reindex(idx_concat).values.astype(float)
        mask = ~np.isnan(real_concat)
        rmse = float(np.sqrt(np.mean((real_concat[mask] - ypred_concat[mask])**2))) if mask.any() else np.nan

        return {
            "model": model,
            "meta": {
                "lookback": lookback,
                "horizon": horizon,
                "exog_dim": (0 if X_monthly is None else X_monthly.shape[1]),
            },
            "idx": idx_concat,
            "y_pred": ypred_concat,
            "RMSE": rmse,
            "train_time": train_time,
        }
    else:
        return {
            "model": model,
            "meta": {
                "lookback": lookback,
                "horizon": horizon,
                "exog_dim": (0 if X_monthly is None else X_monthly.shape[1]),
            },
            "idx": pd.DatetimeIndex([]),
            "y_pred": np.array([]),
            "RMSE": np.nan,
            "train_time": train_time,
        }


# ==========================================
#   F O R E C A S T   (bloques MIMO)
# ==========================================
def _build_feature_row(
    y_hist: np.ndarray,
    X_hist_row: Optional[np.ndarray],
    lookback: int,
) -> np.ndarray:
    if X_hist_row is not None:
        return np.concatenate([y_hist[-lookback:], X_hist_row])
    return y_hist[-lookback:]


def _repeat_last_exog(X_hist: Optional[pd.DataFrame], n_steps: int) -> Optional[pd.DataFrame]:
    if X_hist is None or X_hist.empty:
        return None
    last = X_hist.iloc[-1:].copy()
    rep = pd.concat([last] * n_steps, ignore_index=True)
    start = X_hist.index[-1] + pd.offsets.MonthBegin(1)
    rep.index = pd.date_range(start, periods=n_steps, freq="MS")
    rep.columns = X_hist.columns
    return rep


def forecast_mimo(
    y_monthly_hist: pd.Series,
    X_monthly_hist: Optional[pd.DataFrame],
    model,
    meta: Dict,
    start_forecast: pd.Timestamp,
    X_future: Optional[pd.DataFrame] = None,
    total_h: int = 36,
) -> pd.Series:
    lookback = meta["lookback"]
    horizon = meta["horizon"]

    y = _ensure_ms_index(y_monthly_hist)
    if X_monthly_hist is not None:
        X_monthly_hist = _ensure_ms_index(X_monthly_hist).reindex(y.index)

    cutoff = pd.Timestamp(start_forecast) - pd.offsets.MonthBegin(1)
    y_hist = y.loc[:cutoff].copy()
    if len(y_hist) < lookback:
        raise ValueError("Historia insuficiente para el lookback solicitado.")

    if X_future is not None:
        X_future = _ensure_ms_index(X_future)
    else:
        X_future = _repeat_last_exog(X_monthly_hist, n_steps=total_h)

    preds, idx_out = [], []
    y_buffer = y_hist.values.astype(float)

    if X_future is not None:
        X_future = X_future.reindex(
            pd.date_range(pd.Timestamp(start_forecast), periods=total_h, freq="MS")
        )

    steps_done = 0
    current_start = pd.Timestamp(start_forecast)

    while steps_done < total_h:
        block_h = min(horizon, total_h - steps_done)

        x_exog_row = None
        if X_monthly_hist is not None:
            month_prev = current_start - pd.offsets.MonthBegin(1)
            if month_prev in X_monthly_hist.index:
                x_exog_row = X_monthly_hist.loc[month_prev].values.astype(float)
            elif X_future is not None and month_prev in X_future.index:
                x_exog_row = X_future.loc[month_prev].values.astype(float)
            else:
                x_exog_row = X_monthly_hist.iloc[-1].values.astype(float)

        feat_row = _build_feature_row(
            y_hist=y_buffer, X_hist_row=x_exog_row, lookback=lookback
        ).reshape(1, -1)

        y_block = model.predict(feat_row).ravel()
        y_block = y_block[:block_h]

        months = pd.date_range(current_start, periods=block_h, freq="MS")
        preds.append(y_block)
        idx_out.append(months)

        y_buffer = np.concatenate([y_buffer, y_block])
        steps_done += block_h
        current_start = months[-1] + pd.offsets.MonthBegin(1)

    pred_vals = np.concatenate(preds)
    pred_idx = pd.DatetimeIndex(np.concatenate([idx.values for idx in idx_out]))
    return pd.Series(pred_vals, index=pred_idx, name="forecast")


# ==========================================
#              M A I N
# ==========================================
def run(institucion: int, sucursal: int, templateid: int):
    suc_matriz, suc_dir, plots_dir = crear_carpeta_institucion(institucion, sucursal)
    print("INICIO\n---------------------------------------")

    loader = Loader("./dataset/Crediguate.csv")
    loader.load_data()
    dataset = loader.getDataset()
    print("[OK] CSV leído y preparado por Loader.")

    # 2) Limpieza numérica + asegurar MS
    # IMPORTANTE: NO rellenar con 0 aquí (evita inventar futuro y aplastar la gráfica)
    df_num = dataset.apply(pd.to_numeric, errors="coerce")
    if not isinstance(df_num.index, pd.DatetimeIndex):
        df_num.index = pd.to_datetime(df_num.index)
    df_num = df_num.asfreq("MS").sort_index()

    # ========== CONFIG TEMPORAL ==========
    TRAIN_END = pd.Timestamp("2025-03-01")
    TEST_START = pd.Timestamp("2025-04-01")  # solo para referencia, NO para start_forecast
    HORIZON = 12
    FUTURE_H = 36
    LOOKBACK = 12

    # ========== EXÓGENAS (opcionales) ==========
    exog_names_guess = ["infla_anual", "tipo_cambio_dolar", "cetes_28dias"]
    exog_present = [c for c in exog_names_guess if c in df_num.columns]
    X_m_all = df_num[exog_present] if exog_present else None

    # ========== SELECCIÓN DE CUENTAS ==========
    candidate_targets = [c for c in df_num.columns if c not in exog_present]
    ONLY_TARGETS = None
    targets = ONLY_TARGETS if ONLY_TARGETS else candidate_targets

    # ========== SALIDAS ==========
    xlsx_path = os.path.join(suc_matriz, "predicciones_MIMO.xlsx")
    writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
    resumen_rows = []

    for col in targets:
        try:
            y_m = df_num[col].astype(float)
        except Exception:
            print(f"Saltando {col}: no es numérica.")
            continue

        # 1) Recorta a último dato real (evita "futuro" NaN/0 inventado)
        last_real = y_m.last_valid_index()
        if last_real is None:
            print(f"[{col}] sin datos válidos → se salta.")
            continue
        y_m_real = y_m.loc[:last_real].copy()

        # 2) Para entrenar, rellena huecos SOLO dentro del rango real
        y_m_train = y_m_real.asfreq("MS").fillna(0.0)

        if np.allclose(y_m_train.values, 0.0):
            print(f"[{col}] toda la serie es 0 → se salta.")
            continue

        print(f"\n=== Entrenando {col} (MIMO mensual, horizon={HORIZON}) ===")

        X_m = None
        if X_m_all is not None:
            X_m = X_m_all.loc[y_m_train.index].copy()

        try:
            res = train_mimo_monthly(
                y_monthly=y_m_train,
                X_monthly=X_m,
                train_end=TRAIN_END,
                lookback=LOOKBACK,
                horizon=HORIZON,
                model_type="ridge"
            )
        except Exception as e:
            print(f"Saltando {col}: {e}")
            continue

        # TEST (si hay)
        idx_test = pd.DatetimeIndex(res["idx"])
        real_test = y_m_train.reindex(idx_test)
        pred_test = pd.Series(res["y_pred"], index=idx_test, name="pred_test")
        if len(idx_test):
            print(f"  RMSE TEST ({idx_test[0].date()}→{idx_test[-1].date()}): {res['RMSE']:.4f}")
        else:
            print("  (sin tramo de test)")

        plot_monthly(
            real=real_test,
            pred=pred_test,
            train_end=TRAIN_END,
            title=f"{col} — TEST MIMO {HORIZON}m",
            save_path=os.path.join(plots_dir, f"{col}_TEST_mimo{HORIZON}.png"),
        )

        # ======= FORECAST: 36 meses DESPUÉS del último real =======
        start_forecast = (y_m_real.asfreq("MS").last_valid_index() + pd.offsets.MonthBegin(1))

        forecast_series = forecast_mimo(
            y_monthly_hist=y_m_train,
            X_monthly_hist=X_m,
            model=res["model"],
            meta=res["meta"],
            start_forecast=start_forecast,
            X_future=None,
            total_h=FUTURE_H,
        )

        print(
            f"[DEBUG] {col} forecast len={len(forecast_series)} "
            f"from={forecast_series.index[0].date()} to={forecast_series.index[-1].date()}"
        )

        # Plot Forecast (real SIN futuro inventado)
        full_real = y_m_real.asfreq("MS")
        plot_monthly(
            real=full_real,
            pred=forecast_series,
            train_end=TRAIN_END,
            title=f"{col} — Forecast {FUTURE_H}m (desde {start_forecast.date()})",
            save_path=os.path.join(plots_dir, f"{col}_FORECAST_{FUTURE_H}m.png"),
        )

        # Export por cuenta a Excel (TEST + FORECAST)
        df_out = pd.DataFrame({
            "fecha": idx_test,
            "real_test": real_test.values,
            "pred_test": pred_test.values
        })
        df_out_fore = pd.DataFrame({
            "fecha": forecast_series.index,
            "forecast": forecast_series.values
        })

        sheet_name = str(col)[:31]
        startrow = 0
        if len(df_out) > 0:
            df_out.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)
            startrow = len(df_out) + 2
        df_out_fore.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)

        resumen_rows.append({
            "cuenta": col,
            "rmse_test": res["RMSE"],
            "train_time_s": res["train_time"],
            "test_inicio": (idx_test[0] if len(idx_test) else pd.NaT),
            "test_fin": (idx_test[-1] if len(idx_test) else pd.NaT),
            "forecast_inicio": forecast_series.index[0],
            "forecast_fin": forecast_series.index[-1]
        })

        crear_carpeta_cuenta(suc_matriz, col)
        if sucursal != 0:
            crear_carpeta_cuenta(suc_dir, col)

    if resumen_rows:
        df_resumen = pd.DataFrame(resumen_rows)
        try:
            df_resumen.sort_values("rmse_test", inplace=True)
        except Exception:
            pass
        df_resumen.to_excel(writer, sheet_name="RESUMEN", index=False)

    writer.close()
    print(f"\n[OK] Exportado Excel: {xlsx_path}")
    print(f"[OK] Plots en: {plots_dir}")
    print("\nFIN")


# ========== EJECUCIÓN DIRECTA ==========
if __name__=="__main__":
    # Validar que se pase un argumento entero
    if len(sys.argv) != 4:
        print("Este programa requiere tres enteros (institucionid, sucursalid, templateid): <enteros> <enteros> <enteros>")
        sys.exit(1)

    try:
        institucion = int(sys.argv[1])
        sucursal = int(sys.argv[2])
        templateid = int(sys.argv[3])
    except ValueError:
        print("El parámetro debe ser un número entero.")
        sys.exit(1)
    root_dir = f"./instituciones"
    os.makedirs(root_dir, exist_ok=True)  # crea ./instituciones/ si no existe
    # Llamar a la función principal
    run(institucion,sucursal,templateid)
