import pandas as pd
import numpy as np
import re
from pathlib import Path

# ================== CONFIGURACIÓN ==================
# Entrada: carpeta con los CSV de validación (cada dataset)
carpeta = Path("/media/antonio/ADATA HD710 PRO/Job/plots")
patron = "*.csv"

# Salida: carpeta donde quedarán los CSV por modelo (Linear.csv, RidgePSO.csv, etc.)
salida_dir = Path("modelreview")
salida_dir.mkdir(parents=True, exist_ok=True)

# Top-3 dentro de cada archivo por esta columna (menor = mejor)
COLUMNA_ORDEN = "diff2"
ASCENDING = True  # menor diff = mejor

# Ranking combinado
ARCHIVO_SCORES = "model_scores.csv"
TOP_PRINT = 5

# Conjunto de modelos conocidos (ajústalo a tu lista real)
MODELOS_CONOCIDOS = {
    "linear","linearpso","ridge","ridgepso","lasso","lassopso",
    "rf","rfpso","dt","dtpso","mlp","mlpso","xgboost","xgboostpso"
}
# ===================================================


def normaliza_modelo(s: str) -> str:
    """Quita espacios y baja a lowercase para comparar con MODELOS_CONOCIDOS."""
    return str(s).strip().lower()


def detectar_columna_modelo(df: pd.DataFrame) -> str | None:
    """
    1) Si existe 'modelo' (case-insensitive), úsala.
    2) Si no, busca columna de tipo texto donde >=50% de los valores no nulos
       estén en MODELOS_CONOCIDOS.
    Devuelve el nombre de la columna o None si no encuentra.
    """
    for c in df.columns:
        if str(c).strip().lower() == "modelo":
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for c in obj_cols:
        vals = df[c].dropna().astype(str).map(normaliza_modelo)
        if len(vals) == 0:
            continue
        ratio_known = (vals.isin(MODELOS_CONOCIDOS)).mean()
        if ratio_known >= 0.5:  # umbral flexible
            return c
    return None


def asegurar_columna_orden(df: pd.DataFrame, preferida: str) -> str:
    if preferida in df.columns:
        return preferida
    candidatos = ["diff2","MP2","MR2","diff1","MP1","MR1","diff0","MP0","MR0"]
    for c in candidatos:
        if c in df.columns:
            return c
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[-1] if nums else df.columns[-1]


def columnas_error_disponibles(df: pd.DataFrame) -> list[str]:
    """Prefiere diff*, si no hay usa MR*/MP*."""
    diff_cols = [c for c in df.columns if re.match(r"^diff\d+$", str(c), re.IGNORECASE)]
    if diff_cols:
        return diff_cols
    return [c for c in df.columns if re.match(r"^(MR|MP)\d+$", str(c), re.IGNORECASE)]


def append_top3(df_nuevo: pd.DataFrame, out_path: Path) -> None:
    """Apendea top-3 de un modelo al archivo correspondiente (sin columnas extras)."""
    if out_path.exists():
        df_nuevo.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df_nuevo.to_csv(out_path, index=False, mode="w", header=True)


def generar_top3_por_modelo():
    """Crea/actualiza archivos por modelo (Linear.csv, RidgePSO.csv, etc.)."""
    for csv_path in sorted(carpeta.glob(patron)):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # asegurar numéricas
            for c in df.columns:
                if re.match(r"^(diff|MR|MP)\d+$", str(c), re.IGNORECASE):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            col_modelo = detectar_columna_modelo(df)
            if col_modelo is None:
                print(f"⚠️ {csv_path.name}: no se detectó la columna de modelo; se omite.")
                continue

            col_orden = asegurar_columna_orden(df, COLUMNA_ORDEN)

            # agrupar por modelo y tomar top-3 (dentro de ESTE archivo)
            for modelo_val, grupo in df.groupby(col_modelo, dropna=False):
                mod_norm = normaliza_modelo(modelo_val)
                if mod_norm not in MODELOS_CONOCIDOS:
                    # esto evita crear archivos con nombres 'validacionmodelo_xxx'
                    continue

                top3 = grupo.sort_values(by=col_orden, ascending=ASCENDING).head(3).copy()
                out_path = salida_dir / f"{modelo_val}.csv"  # usa el texto original como nombre
                append_top3(top3, out_path)

            print(f"✓ {csv_path.name}: top-3 por '{col_orden}'")

        except Exception as e:
            print(f"⚠️ Error leyendo {csv_path.name}: {e}")


def calcular_scores_por_modelo() -> pd.DataFrame:
    """
    Lee solo archivos cuyo nombre sea un modelo conocido.
    Calcula:
      - frecuencia = # de filas (cuántas veces apareció entre top-3)
      - promedio_error = MAE sobre diff* (o MR*/MP* si no hay diff*)
    """
    registros = []
    for path_csv in sorted(salida_dir.glob("*.csv")):
        # filtra por nombre de archivo que sea un modelo conocido
        stem_norm = normaliza_modelo(path_csv.stem)
        if stem_norm not in MODELOS_CONOCIDOS:
            continue

        try:
            df = pd.read_csv(path_csv)
            if df.empty:
                continue

            cols_err = columnas_error_disponibles(df)
            if not cols_err:
                continue

            for c in cols_err:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            vals = df[cols_err].to_numpy(dtype=float)
            mae = np.nanmean(np.abs(vals))  # MAE global sobre diff*/MR*/MP*

            registros.append({
                "modelo": path_csv.stem,       # nombre limpio del archivo = nombre del modelo
                "frecuencia": int(df.shape[0]),
                "promedio_error": float(mae),
                "columnas_usadas": ",".join(cols_err)
            })
        except Exception as e:
            print(f"⚠️ Error procesando {path_csv.name}: {e}")

    return pd.DataFrame(registros)


def ranking_combinado(scores: pd.DataFrame) -> pd.DataFrame:
    """Ordena por frecuencia (desc) y luego por error promedio (asc)."""
    return scores.sort_values(
        by=["frecuencia","promedio_error"],
        ascending=[False, True],
        ignore_index=True
    )


def elegir_mejor_modelo(scores_rank: pd.DataFrame) -> dict:
    """De los más frecuentes, el de menor error promedio."""
    if scores_rank.empty:
        return {}
    max_freq = scores_rank["frecuencia"].max()
    top = scores_rank[scores_rank["frecuencia"] == max_freq]
    ganador = top.sort_values(by="promedio_error").iloc[0]
    return {
        "ganador_modelo": ganador["modelo"],
        "ganador_frecuencia": int(ganador["frecuencia"]),
        "ganador_promedio_error": float(ganador["promedio_error"]),
        "grupo_mas_frecuente": top.sort_values(by="promedio_error").reset_index(drop=True)
    }


def main():
    # 1) Genera/actualiza archivos por modelo (solo si el CSV tiene columna de modelo válida)
    generar_top3_por_modelo()

    # 2) Calcula scores por modelo leyendo SOLO archivos cuyos nombres sean modelos conocidos
    scores = calcular_scores_por_modelo()
    if scores.empty:
        print("⚠️ No se encontraron CSV por modelo válidos en la carpeta de salida.")
        return

    # 3) Ranking combinado
    scores_rank = ranking_combinado(scores)
    out_path = salida_dir / ARCHIVO_SCORES
    scores_rank.to_csv(out_path, index=False)
    print(f"\n✅ Ranking combinado guardado en: {out_path}")

    # 4) Muestra top
    print("\n🏆 TOP (frecuencia ↓, error promedio ↑)")
    print(scores_rank.head(TOP_PRINT).to_string(index=False))

    # 5) Ganador: de los más frecuentes, el de menor error
    res = elegir_mejor_modelo(scores_rank)
    if res:
        print("\n⭐ Ganador según tu criterio:")
        print(f"   - Modelo: {res['ganador_modelo']}")
        print(f"   - Frecuencia (máxima): {res['ganador_frecuencia']}")
        print(f"   - Promedio de error (MAE): {res['ganador_promedio_error']:.6f}")

        print("\n📊 Modelos en el grupo MÁS FRECUENTE (ordenados por menor error):")
        print(res["grupo_mas_frecuente"].to_string(index=False))
    else:
        print("⚠️ No fue posible determinar un ganador.")

if __name__ == "__main__":
    main()
