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
    output_path: str = "predicciones_todas.xlsx",
    meses_objetivo: int = 15,
    real_desde: str = "2024-02-01",
):
    # --- 1) normalizar real ---
    df_real = real_full.copy()
    if not isinstance(df_real.index, pd.DatetimeIndex):
        df_real.index = pd.to_datetime(df_real.index)
    df_real = df_real.asfreq("MS")

    if df_real.shape[1] != 1:
        raise ValueError("real_full debe tener solo 1 columna.")
    real_colname = df_real.columns[0]
    df_real = df_real.rename(columns={real_colname: "real"})

    real_desde = pd.to_datetime(real_desde)
    df_real_recorte = df_real.loc[real_desde:].copy()

    # --- 2) armar candidatos desde model_scores ---
    candidatos = []
    for model_name, raw_res in model_scores.items():
        norm = normalizar_resultado_para_export(
            raw_res,
            df_real,
            meses_objetivo=meses_objetivo,
        )
        y_true = np.asarray(norm["y_true"], float)
        y_pred = np.asarray(norm["y_pred"], float)
        idx    = pd.to_datetime(norm["idx"])

        if len(idx) == 0:
            continue

        # métricas robustas
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        mask_nz = np.abs(y_true) > 1e-6
        if mask_nz.any():
            mape = float(
                np.mean(np.abs((y_true[mask_nz] - y_pred[mask_nz]) / y_true[mask_nz])) * 100
            )
        else:
            mape = np.nan

        smape = float(
            np.mean(
                2.0 * np.abs(y_pred - y_true) /
                (np.abs(y_true) + np.abs(y_pred) + 1e-6)
            ) * 100
        )

        candidatos.append({
            "modelo": model_name,
            "norm": norm,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "SMAPE": smape,
        })

    # si no hay modelos válidos, igual insertamos una fila vacía
    if not candidatos:
        best_model_name = None
        df_pred = pd.DataFrame(columns=["cuenta","modelo","fecha","real","prediccion"])
        df_resumen = pd.DataFrame([{
            "cuenta": cuenta,
            "modelo": None,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_%": np.nan,
            "SMAPE_%": np.nan,
            "similitud_%": np.nan,
        }])
    else:
        # --- 3) escoger el mejor por RMSE ---
        candidatos.sort(key=lambda d: d["RMSE"])
        best = candidatos[0]

        best_model_name = best["modelo"]
        best_norm       = best["norm"]
        y_true          = np.asarray(best_norm["y_true"], float)
        y_pred          = np.asarray(best_norm["y_pred"], float)
        idx             = pd.to_datetime(best_norm["idx"])
        mae             = best["MAE"]
        rmse            = best["RMSE"]
        mape            = best["MAPE"]
        smape           = best["SMAPE"]

        # similitud
        mask_nz = np.abs(y_true) > 1e-6
        if mask_nz.any():
            base = float(np.mean(np.abs(y_true[mask_nz])))
            if base < 1.0:
                base = 1.0
            similitud = 100.0 * max(0.0, 1.0 - (mae / base))
        else:
            # real todo cero
            similitud = similitud_cuando_real_cero(y_pred, tol=1000.0)

        similitud = float(min(100.0, max(0.0, similitud)))

        df_pred = pd.DataFrame({
            "cuenta": cuenta,
            "modelo": best_model_name,
            "fecha": idx.date,
            "real": y_true,
            "prediccion": y_pred,
        })

        df_resumen = pd.DataFrame([{
            "cuenta": cuenta,
            "modelo": best_model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_%": mape,
            "SMAPE_%": smape,
            "similitud_%": similitud,
        }])

    # ================== APPEND AL EXCEL ==================
    # si no existe, creamos desde cero
    if not os.path.exists(output_path):
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # hoja real: solo la de esta cuenta
            real_reset = df_real_recorte.reset_index()
            fecha_col = [c for c in real_reset.columns if c != "real"][0]
            real_reset = real_reset.rename(columns={fecha_col: "fecha"})
            real_reset["fecha"] = pd.to_datetime(real_reset["fecha"]).dt.date
            real_reset.to_excel(writer, sheet_name="real", index=False)

            df_pred.to_excel(writer, sheet_name="predicciones", index=False)
            df_resumen.to_excel(writer, sheet_name="resumen", index=False)
        print(f"[OK] Archivo creado en {output_path} con cuenta {cuenta}")
    else:
        # existe: lo leemos y concatenamos
        xl = pd.ExcelFile(output_path)
        # real: lo dejamos como estaba, no concatenamos real de todas las cuentas para no mezclar
        df_pred_exist = xl.parse("predicciones") if "predicciones" in xl.sheet_names else pd.DataFrame()
        df_res_exist  = xl.parse("resumen")      if "resumen" in xl.sheet_names else pd.DataFrame()

        df_pred_total = pd.concat([df_pred_exist, df_pred], ignore_index=True)
        df_res_total  = pd.concat([df_res_exist,  df_resumen], ignore_index=True)

        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            # volvemos a escribir la hoja real original que había en el archivo
            if "real" in xl.sheet_names:
                df_real_old = xl.parse("real")
                df_real_old.to_excel(writer, sheet_name="real", index=False)
            else:
                # si no había, escribimos la de la última cuenta
                real_reset = df_real_recorte.reset_index()
                fecha_col = [c for c in real_reset.columns if c != "real"][0]
                real_reset = real_reset.rename(columns={fecha_col: "fecha"})
                real_reset["fecha"] = pd.to_datetime(real_reset["fecha"]).dt.date
                real_reset.to_excel(writer, sheet_name="real", index=False)

            df_pred_total.to_excel(writer, sheet_name="predicciones", index=False)
            df_res_total.to_excel(writer, sheet_name="resumen", index=False)

        print(f"[OK] Archivo actualizado en {output_path} con cuenta {cuenta}")