import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import normalizar_resultado_para_export
import os

def similitud_cuando_real_cero(y_pred, tol=50.0):
    mae = float(np.mean(np.abs(y_pred)))
    if mae >= tol:
        return 0.0
    return float(100.0 * (1.0 - mae / tol))

def exportar_predicciones_y_resumen(cuenta: str,
                                    real_full: pd.DataFrame,
                                    model_scores: dict,
                                    output_path: str = None,
                                    meses_objetivo: int = 15):
    # normalizar real
    df_real = real_full.copy()
    if not isinstance(df_real.index, pd.DatetimeIndex):
        df_real.index = pd.to_datetime(df_real.index)
    df_real = df_real.asfreq("MS")

    if df_real.shape[1] != 1:
        raise ValueError("real_full debe tener solo 1 columna.")
    df_real = df_real.rename(columns={df_real.columns[0]: "real"})

    filas_pred = []
    filas_resumen = []

    for model_name, raw_res in model_scores.items():
        norm = normalizar_resultado_para_export(
            raw_res,
            df_real,
            meses_objetivo=meses_objetivo
        )
        y_true = norm["y_true"]
        y_pred = norm["y_pred"]
        idx    = norm["idx"]

        if len(idx) == 0:
            # modelo sin datos útiles
            filas_resumen.append({
                "cuenta": cuenta,
                "modelo": model_name,
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "n_puntos": 0,
            })
            continue

        # detalle
        df_det = pd.DataFrame({
            "cuenta": cuenta,
            "modelo": model_name,
            "fecha": idx,
            "real": y_true,
            "prediccion": y_pred,
        })
        filas_pred.append(df_det)

        # métricas
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        eps = 1e-6
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

        filas_resumen.append({
            "cuenta": cuenta,
            "modelo": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "n_puntos": len(idx),
        })

    df_predicciones = pd.concat(filas_pred, ignore_index=True) if filas_pred else pd.DataFrame()
    df_resumen = pd.DataFrame(filas_resumen)

    if output_path is None:
        output_path = f"predicciones_{cuenta}.xlsx"

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # hoja con real completo
        df_real.reset_index().rename(columns={"index": "fecha"}).to_excel(
            writer, sheet_name="real", index=False
        )
        df_predicciones.to_excel(writer, sheet_name="predicciones", index=False)
        df_resumen.to_excel(writer, sheet_name="resumen", index=False)

    print(f"[OK] Archivo escrito en {output_path}")


def exportar_predicciones_y_resumen_solo_mejor(
    cuenta: str,
    real_full: pd.DataFrame,
    model_scores: dict,
    output_path: str = "predicciones_.xlsx",
    meses_objetivo: int = 15,
    real_desde: str = "2024-02-01",
):
    """
    Guarda TODO en un solo Excel acumulado:
      - hoja 'real': todas las cuentas, todas las fechas (desde real_desde)
      - hoja 'predicciones': solo el MEJOR modelo de esa cuenta
      - hoja 'resumen': solo el MEJOR modelo de esa cuenta

    Si el archivo ya existe, agrega filas.
    Si está corrupto, lo recrea.
    """
    # 1) normalizar real --------------------------------------------------------
    df_real = real_full.copy()
    if not isinstance(df_real.index, pd.DatetimeIndex):
        df_real.index = pd.to_datetime(df_real.index)
    df_real = df_real.asfreq("MS")  # 1 de cada mes

    if df_real.shape[1] != 1:
        raise ValueError("real_full debe tener solo 1 columna.")

    real_desde = pd.to_datetime(real_desde)
    df_real_recorte = df_real.loc[real_desde:].copy()

    # esto será lo que escribimos en la hoja "real" (agregando cuenta)
    df_real_nuevo = (
        df_real_recorte
        .reset_index()
        .rename(columns={df_real_recorte.index.name or "index": "fecha",
                         df_real_recorte.columns[0]: "real"})
    )
    # dejar solo fecha sin hora
    df_real_nuevo["fecha"] = pd.to_datetime(df_real_nuevo["fecha"]).dt.date
    df_real_nuevo.insert(0, "cuenta", cuenta)

    # 2) elegir mejor modelo ----------------------------------------------------
    candidatos = []
    for model_name, raw_res in model_scores.items():
        # normalización flexible: puede venir con 'idx', o solo y_true/y_pred
        idx = raw_res.get("idx", None)
        y_true = raw_res.get("y_true", None)
        y_pred = raw_res.get("y_pred", None)

        if y_true is None or y_pred is None:
            continue

        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        # si no hay fechas, inventamos las últimas len(y_true) del real
        if idx is None:
            if len(df_real) >= len(y_true):
                idx = df_real.index[-len(y_true):]
            else:
                # fallback
                idx = pd.date_range(start="2000-01-01", periods=len(y_true), freq="MS")
        else:
            idx = pd.to_datetime(idx)

        # métrica
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        eps = 1e-6
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

        candidatos.append({
            "modelo": model_name,
            "idx": idx,
            "y_true": y_true,
            "y_pred": y_pred,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })

    # si no hay candidatos (ej. nada se entrenó), igual escribimos la parte real
    if not candidatos:
        _escribir_excel_acumulando(
            output_path,
            df_real_nuevo=df_real_nuevo,
            df_pred_nuevo=pd.DataFrame(),
            df_resumen_nuevo=pd.DataFrame([{
                "cuenta": cuenta,
                "modelo": None,
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "similitud_%": np.nan,
            }]),
        )
        print(f"[OK] Archivo escrito en {output_path} (sin modelos válidos)")
        return

    # 3) escoger mejor por RMSE -------------------------------------------------
    candidatos.sort(key=lambda d: d["RMSE"])
    best = candidatos[0]

    idx    = best["idx"]
    y_true = best["y_true"]
    y_pred = best["y_pred"]
    mae    = best["MAE"]
    rmse   = best["RMSE"]
    mape   = best["MAPE"]
    best_model_name = best["modelo"]

    # similitud robusta
    mean_abs_real = float(np.mean(np.abs(y_true)))
    if mean_abs_real < 1e-6:
        # caso todo-cero: mido qué tan cerca de 0 están las predicciones
        tol = 1.0
        max_err = float(np.max(np.abs(y_pred)))
        similitud = 100.0 * max(0.0, 1.0 - max_err / tol)
    else:
        rel_err = mae / (mean_abs_real + 1e-6)
        similitud = 100.0 * max(0.0, 1.0 - rel_err)
    similitud = float(min(100.0, max(0.0, similitud)))

    # 4) df_pred y df_resumen para ESTE modelo ---------------------------------
    df_pred_nuevo = pd.DataFrame({
        "cuenta": cuenta,
        "modelo": best_model_name,
        "fecha": pd.to_datetime(idx).date,
        "real": y_true,
        "prediccion": y_pred,
    })

    df_resumen_nuevo = pd.DataFrame([{
        "cuenta": cuenta,
        "modelo": best_model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "similitud_%": similitud,
    }])

    # 5) escribir / acumular ----------------------------------------------------
    _escribir_excel_acumulando(
        output_path,
        df_real_nuevo=df_real_nuevo,
        df_pred_nuevo=df_pred_nuevo,
        df_resumen_nuevo=df_resumen_nuevo,
    )

    print(f"[OK] Archivo escrito en {output_path} con el mejor modelo '{best_model_name}'")


def _escribir_excel_acumulando(
    output_path: str,
    df_real_nuevo: pd.DataFrame,
    df_pred_nuevo: pd.DataFrame,
    df_resumen_nuevo: pd.DataFrame,
):
    """
    Escribe/append a un Excel existente.
    Si el archivo está corrupto o no existe, lo crea desde cero.
    Siempre escribe 3 hojas: real, predicciones, resumen.
    """
    if not os.path.exists(output_path):
        # crear desde cero
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            df_real_nuevo.to_excel(writer, sheet_name="real", index=False)
            df_pred_nuevo.to_excel(writer, sheet_name="predicciones", index=False)
            df_resumen_nuevo.to_excel(writer, sheet_name="resumen", index=False)
        return

    # si existe, intentamos leerlo
    try:
        existing_real = pd.read_excel(output_path, sheet_name="real")
        existing_pred = pd.read_excel(output_path, sheet_name="predicciones")
        existing_res  = pd.read_excel(output_path, sheet_name="resumen")

        real_out = pd.concat([existing_real, df_real_nuevo], ignore_index=True)
        pred_out = pd.concat([existing_pred, df_pred_nuevo], ignore_index=True)
        res_out  = pd.concat([existing_res,  df_resumen_nuevo], ignore_index=True)

        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            real_out.to_excel(writer, sheet_name="real", index=False)
            pred_out.to_excel(writer, sheet_name="predicciones", index=False)
            res_out.to_excel(writer, sheet_name="resumen", index=False)
    except Exception:
        # archivo corrupto → lo recreamos
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            df_real_nuevo.to_excel(writer, sheet_name="real", index=False)
            df_pred_nuevo.to_excel(writer, sheet_name="predicciones", index=False)
            df_resumen_nuevo.to_excel(writer, sheet_name="resumen", index=False)