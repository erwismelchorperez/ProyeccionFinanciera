import pandas as pd
from pathlib import Path
import re
import numpy as np

# ================== CONFIGURACIÓN ==================
# Carpeta con los CSV de entrada (los "validacionmodelo_*.csv" por archivo)
carpeta = Path("/media/antonio/ADATA HD710 PRO/Job/plots")

# Carpeta donde se guardarán los CSV por modelo (Linear.csv, RidgePSO.csv, etc.)
salida_dir = Path("modelreview")

# Patrón de archivos a leer desde 'carpeta'
patron = "*.csv"

# Top-3 dentro de cada archivo, por esta columna:
#   - Para "menor error es mejor": columna_orden = "diff2", ascending = True
#   - Para "mayor métrica es mejor": columna_orden = "MP2",   ascending = False
columna_orden = "diff2"
ascending = True

# Nombre del archivo de resumen de promedios por modelo
archivo_scores = "model_scores.csv"
# ===================================================


def detectar_columna_modelo(df: pd.DataFrame) -> str:
    """Detecta la columna que contiene el nombre del modelo."""
    for c in df.columns:
        if str(c).strip().lower() == "modelo":
            return c
    # Si no se encuentra, intenta con la segunda columna (como en tus ejemplos)
    return df.columns[1] if len(df.columns) >= 2 else df.columns[0]


def asegurar_columna_orden(df: pd.DataFrame, preferida: str) -> str:
    """Asegura que la columna para ordenar exista; si no, busca alternativas."""
    if preferida in df.columns:
        return preferida
    candidatos = ["diff2", "MP2", "MR2", "diff1", "MP1", "MR1", "diff0", "MP0", "MR0"]
    for c in candidatos:
        if c in df.columns:
            return c
    # Último recurso: la última columna numérica
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[-1] if nums else df.columns[-1]


def append_con_encabezado_unico(df: pd.DataFrame, out_path: Path) -> None:
    """Escribe/apende un CSV garantizando encabezado único."""
    if out_path.exists():
        df.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df.to_csv(out_path, index=False, mode="w", header=True)


def generar_top3_por_modelo(carpeta: Path, salida_dir: Path, patron: str,
                            columna_orden: str, ascending: bool) -> None:
    """Lee todos los CSV de entrada y escribe top-3 por modelo en archivos por modelo."""
    salida_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(carpeta.glob(patron)):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            col_modelo = detectar_columna_modelo(df)
            col_ord = asegurar_columna_orden(df, columna_orden)

            # Asegurar tipo numérico en columnas numéricas conocidas
            for c in df.columns:
                if re.match(r"^(diff|MR|MP)\d+$", str(c), flags=re.IGNORECASE):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Para cada modelo dentro del archivo, tomar top-3 por col_ord
            for modelo, grupo in df.groupby(col_modelo, dropna=False):
                grupo_ordenado = grupo.sort_values(by=col_ord, ascending=ascending).head(3).copy()
                # Inserta el origen (archivo) si te sirve
                grupo_ordenado.insert(0, "archivo_origen", csv_path.stem)

                out_path = salida_dir / f"{str(modelo).strip()}.csv"
                append_con_encabezado_unico(grupo_ordenado, out_path)

            print(f"✓ {csv_path.name}: top-3 por '{col_ord}' (ascending={ascending})")

        except Exception as e:
            print(f"⚠️ Error leyendo {csv_path}: {e}")


def columnas_error_disponibles(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas a usar como 'error' en este orden de preferencia:
    1) diff*   2) MR* / MP*
    """
    cols = [c for c in df.columns if re.match(r"^diff\d+$", str(c), flags=re.IGNORECASE)]
    if cols:
        return cols
    cols = [c for c in df.columns if re.match(r"^(MR|MP)\d+$", str(c), flags=re.IGNORECASE)]
    return cols


def resumir_promedios_por_modelo(salida_dir: Path,
                                 usar_absoluto: bool = True,
                                 archivo_salida: str = "model_scores.csv",
                                 top_n: int = 3) -> pd.DataFrame:
    registros = []

    for path_csv in sorted(salida_dir.glob("*.csv")):
        if path_csv.name in {archivo_salida, "dataset.csv", "dataset_top3.csv"}:
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
            if usar_absoluto:
                vals = np.abs(vals)

            promedio_error = np.nanmean(vals)

            registros.append({
                "modelo": path_csv.stem,
                "promedio_error": promedio_error,
                "num_filas": int(df.shape[0]),   # frecuencia = cuántas veces apareció
                "columnas_usadas": ",".join(cols_err)
            })

        except Exception as e:
            print(f"⚠️ Error procesando {path_csv.name}: {e}")

    if not registros:
        print("⚠️ No se encontraron datasets por modelo.")
        return pd.DataFrame()

    scores = pd.DataFrame(registros)

    # Ordenar por error promedio (menor = mejor)
    scores = scores.sort_values(by="promedio_error", ascending=True, ignore_index=True)

    # Guardar
    out_path = salida_dir / archivo_salida
    scores.to_csv(out_path, index=False)
    print(f"\n✅ Promedios por modelo guardados en: {out_path}")

    # Mostrar resultados
    print("\n🏆 Top 3 modelos (menor error promedio):")
    print(scores.head(top_n).to_string(index=False))

    # Modelo con menor error promedio
    mejor_modelo = scores.iloc[0]["modelo"]
    print(f"\n⭐ Mejor modelo por error promedio: {mejor_modelo}")

    # Modelo más frecuente
    modelo_mas_usado = scores.sort_values(by="num_filas", ascending=False).iloc[0]
    print(f"📊 Modelo más usado (frecuente): {modelo_mas_usado['modelo']} "
          f"({modelo_mas_usado['num_filas']} apariciones)")

    return scores


if __name__ == "__main__":
    # 1) Generar/actualizar los CSV por modelo con el top-3 por archivo
    
    generar_top3_por_modelo(
        carpeta=carpeta,
        salida_dir=salida_dir,
        patron=patron,
        columna_orden=columna_orden,
        ascending=ascending
    )
    
    
    # 2) Resumir promedios por modelo (MEJOR = menor promedio de error absoluto)
    _ = resumir_promedios_por_modelo(
        salida_dir=salida_dir,
        usar_absoluto=True,               # MAE (recomendado)
        archivo_salida=archivo_scores,
        top_n=3
    )
