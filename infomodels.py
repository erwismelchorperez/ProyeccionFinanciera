import pandas as pd
import numpy as np
import re
from pathlib import Path

# ====== CONFIGURACIÓN ======
carpeta = Path("/media/antonio/ADATA HD710 PRO/Job/plots")  # carpeta con los CSV de entrada
patron = "*.csv"
archivo_scores = "review.csv"
salida = Path("modelreview") / archivo_scores
# ===========================

def detectar_columna_modelo(df: pd.DataFrame) -> str:
    # Intenta 'modelo' (case-insensitive); si no existe, toma la segunda columna.
    for c in df.columns:
        if str(c).strip().lower() == "modelo":
            return c
    return df.columns[1] if len(df.columns) >= 2 else df.columns[0]

def cols_por_patron(df: pd.DataFrame, patron_regex: str):
    return [c for c in df.columns if re.match(patron_regex, str(c), flags=re.IGNORECASE)]

def asegurar_numericas(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def puntaje_error(df: pd.DataFrame) -> pd.Series:
    """
    Retorna una Serie 'score' por fila:
    1) diff2 si existe;
    2) si no, promedio de |diff*|;
    3) si no, promedio de |MR* y MP*|.
    """
    # 1) diff2 directo
    if "diff2" in df.columns:
        return df["diff2"].abs()

    # 2) promedio de |diff*|
    diff_cols = cols_por_patron(df, r"^diff\d+$")
    if diff_cols:
        asegurar_numericas(df, diff_cols)
        return df[diff_cols].abs().mean(axis=1)

    # 3) promedio de |MR* y MP*|
    mr_cols = cols_por_patron(df, r"^MR\d+$")
    mp_cols = cols_por_patron(df, r"^MP\d+$")
    cand = mr_cols + mp_cols
    if cand:
        asegurar_numericas(df, cand)
        return df[cand].abs().mean(axis=1)

    # Si no hay nada numérico relacionado, todo a NaN (no debería pasar)
    return pd.Series(np.nan, index=df.index)

def ordenar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    # Orden objetivo; solo incluimos las que existan
    orden = ["nombre", "modelo", "MR0", "MP0", "diff0", "MR1", "MP1", "diff1", "MR2", "MP2", "diff2"]
    cols_presentes = [c for c in orden if c in df.columns]
    # agrega cualquier otra columna no listada (al final), por si acaso
    restantes = [c for c in df.columns if c not in cols_presentes]
    return df[cols_presentes + restantes]

def main():
    filas_resumen = []
    for csv_path in sorted(carpeta.glob(patron)):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Detecta columna de modelo (por si la necesitas en salida)
            col_modelo = detectar_columna_modelo(df)
            if col_modelo not in df.columns:
                # Garantiza que exista alguna columna 'modelo' en salida
                df = df.rename(columns={col_modelo: "modelo"})
                col_modelo = "modelo"

            # Calcula score de error y toma la mejor fila (mínimo)
            score = puntaje_error(df)
            idx_mejor = score.idxmin()
            fila = df.loc[idx_mejor].copy()

            # Asegura que 'modelo' exista con ese nombre
            if "modelo" not in fila.index and col_modelo in fila.index:
                fila.rename({col_modelo: "modelo"}, inplace=True)

            # Inserta nombre de archivo (sin extensión)
            fila = fila.to_dict()
            fila["nombre"] = csv_path.stem

            filas_resumen.append(fila)

        except Exception as e:
            print(f"⚠️ Error leyendo {csv_path.name}: {e}")

    if not filas_resumen:
        print("⚠️ No se pudieron generar filas de resumen.")
        return

    resumen = pd.DataFrame(filas_resumen)

    # Renombra a 'modelo' si quedó con mayúsculas diferentes
    if "modelo" not in resumen.columns:
        posibles = [c for c in resumen.columns if str(c).strip().lower() == "modelo"]
        if posibles:
            resumen.rename(columns={posibles[0]: "modelo"}, inplace=True)

    # Ordena columnas con el encabezado solicitado (omitiendo las que no existan)
    resumen = ordenar_columnas(resumen)

    # Guarda CSV con encabezado único
    resumen.to_csv(salida, index=False)
    print(f"✅ Resumen guardado en: {salida}")

    # Muestra una vista previa
    print("\nPrimeras filas del resumen:")
    print(resumen.head().to_string(index=False))

if __name__ == "__main__":
    main()
