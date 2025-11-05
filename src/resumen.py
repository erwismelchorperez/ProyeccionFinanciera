import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import normalizar_resultado_para_export
import os

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
    output_path: str = None,
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
    # columna real
    real_colname = df_real.columns[0]
    df_real = df_real.rename(columns={real_colname: "real"})

    # recorte de la hoja "real" (solo para mostrar en excel, si quisieras)
    real_desde = pd.to_datetime(real_desde)
    df_real_recorte = df_real.loc[real_desde:].copy()

    # --- 2) recorrer modelos y normalizar resultados ---
    candidatos = []
    for model_name, raw_res in model_scores.items():
        norm = normalizar_resultado_para_export(
            raw_res,
            df_real,
            meses_objetivo=meses_objetivo,
        )
        y_true = norm["y_true"]
        y_pred = norm["y_pred"]
        idx    = pd.to_datetime(norm["idx"])

        if len(idx) == 0:
            continue

        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        eps  = 1e-6
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

        candidatos.append({
            "modelo": model_name,
            "norm": norm,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })

    # nombre de salida
    if output_path is None:
        output_path = f"predicciones_{cuenta}.xlsx"

    # --- 3) si no hay candidatos, escribir vacío pero SIN borrar lo que ya hay ---
    if not candidatos:
        # si ya existe, solo agregamos una línea al resumen
        if os.path.exists(output_path):
            # leemos lo que ya está
            with pd.ExcelFile(output_path, engine="openpyxl") as xls:
                try:
                    df_pred_old = pd.read_excel(xls, sheet_name="predicciones")
                except Exception:
                    df_pred_old = pd.DataFrame()
                try:
                    df_res_old = pd.read_excel(xls, sheet_name="resumen")
                except Exception:
                    df_res_old = pd.DataFrame()

            nuevo_resumen = pd.DataFrame([{
                "cuenta": cuenta,
                "modelo": None,
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE_%": np.nan,
                "similitud_%": np.nan,
            }])

            df_res_out = pd.concat([df_res_old, nuevo_resumen], ignore_index=True)

            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                # dejamos las otras hojas como estaban
                if not df_pred_old.empty:
                    df_pred_old.to_excel(writer, sheet_name="predicciones", index=False)
                df_res_out.to_excel(writer, sheet_name="resumen", index=False)
            print(f"[OK] Archivo actualizado en {output_path} (sin modelos válidos)")
            return
        else:
            # no existía, lo creamos mínimo con resumen
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # hoja real (si quieres guardarla)
                real_reset = df_real_recorte.reset_index()
                fecha_col = [c for c in real_reset.columns if c != "real"][0]
                real_reset = real_reset.rename(columns={fecha_col: "fecha"})
                real_reset["fecha"] = pd.to_datetime(real_reset["fecha"]).dt.date
                real_reset.to_excel(writer, sheet_name="real", index=False)

                pd.DataFrame().to_excel(writer, sheet_name="predicciones", index=False)
                pd.DataFrame([{
                    "cuenta": cuenta,
                    "modelo": None,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE_%": np.nan,
                    "similitud_%": np.nan,
                }]).to_excel(writer, sheet_name="resumen", index=False)
            print(f"[OK] Archivo escrito en {output_path} (sin modelos válidos)")
            return

    # --- 4) escoger mejor por RMSE ---
    candidatos.sort(key=lambda d: d["RMSE"])
    best = candidatos[0]

    best_model_name = best["modelo"]
    best_norm       = best["norm"]
    y_true          = best_norm["y_true"]
    y_pred          = best_norm["y_pred"]
    idx             = pd.to_datetime(best_norm["idx"])

    mae   = best["MAE"]
    rmse  = best["RMSE"]
    mape  = best["MAPE"]

    # similitud robusta
    eps = 1e-6
    mean_abs_real = float(np.mean(np.abs(y_true)))
    if mean_abs_real < 1e-6:
        tol = 1.0
        max_err = float(np.max(np.abs(y_pred)))
        similitud = 100.0 * max(0.0, 1.0 - max_err / tol)
    else:
        rel_err = mae / (mean_abs_real + eps)
        similitud = 100.0 * max(0.0, 1.0 - rel_err)
    similitud = float(min(100.0, max(0.0, similitud)))

    # --- 5) df_pred solo con el mejor ---
    df_pred_new = pd.DataFrame({
        "cuenta": cuenta,
        "modelo": best_model_name,
        "fecha": idx.date,     # sin hora
        "real": y_true,
        "prediccion": y_pred,
    })

    # --- 6) df_resumen solo con el mejor ---
    df_resumen_new = pd.DataFrame([{
        "cuenta": cuenta,
        "modelo": best_model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape,
        "similitud_%": similitud,
    }])

    # --- 7) escribir / append ---
    if os.path.exists(output_path):
        # leemos lo existente
        with pd.ExcelFile(output_path, engine="openpyxl") as xls:
            try:
                df_pred_old = pd.read_excel(xls, sheet_name="predicciones")
            except Exception:
                df_pred_old = pd.DataFrame()
            try:
                df_res_old = pd.read_excel(xls, sheet_name="resumen")
            except Exception:
                df_res_old = pd.DataFrame()
            try:
                df_real_old = pd.read_excel(xls, sheet_name="real")
            except Exception:
                df_real_old = pd.DataFrame()

        # concatenamos
        df_pred_out = pd.concat([df_pred_old, df_pred_new], ignore_index=True)
        df_res_out  = pd.concat([df_res_old, df_resumen_new], ignore_index=True)

        # escribimos reemplazando las hojas
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            # si ya había 'real', la volvemos a escribir tal cual
            if not df_real_old.empty:
                df_real_old.to_excel(writer, sheet_name="real", index=False)
            else:
                # si no había, escribimos el recorte actual
                real_reset = df_real_recorte.reset_index()
                fecha_col = [c for c in real_reset.columns if c != "real"][0]
                real_reset = real_reset.rename(columns={fecha_col: "fecha"})
                real_reset["fecha"] = pd.to_datetime(real_reset["fecha"]).dt.date
                real_reset.to_excel(writer, sheet_name="real", index=False)

            df_pred_out.to_excel(writer, sheet_name="predicciones", index=False)
            df_res_out.to_excel(writer, sheet_name="resumen", index=False)
    else:
        # archivo nuevo
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            real_reset = df_real_recorte.reset_index()
            fecha_col = [c for c in real_reset.columns if c != "real"][0]
            real_reset = real_reset.rename(columns={fecha_col: "fecha"})
            real_reset["fecha"] = pd.to_datetime(real_reset["fecha"]).dt.date
            real_reset.to_excel(writer, sheet_name="real", index=False)

            df_pred_new.to_excel(writer, sheet_name="predicciones", index=False)
            df_resumen_new.to_excel(writer, sheet_name="resumen", index=False)

    print(f"[OK] Archivo actualizado en {output_path} con el mejor modelo '{best_model_name}'")