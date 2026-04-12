import pandas as pd

# 1. Leer el archivo CSV
df = pd.read_csv("./dataset/Crediguate_actualizado.csv")

niveles = df["NIVEL"].tolist()
n = len(niveles)

# Inicializamos la nueva columna con 'NO'
proyeccion = ["NO"] * n

# 2. Detectar segmentos donde NIVEL no disminuye
inicio_segmento = 0
for i in range(1, n + 1):
    if i == n or niveles[i] < niveles[i - 1]:
        segmento = niveles[inicio_segmento:i]
        max_val = max(segmento)

        for j in range(inicio_segmento, i):
            if niveles[j] == max_val:
                proyeccion[j] = "SI"

        inicio_segmento = i

# 3. Insertar la columna como la cuarta columna (índice 3)
df.insert(3, "proyeccion", proyeccion)

# 4. Rellenar con 0 todas las columnas después de la columna 'proyeccion'
proy_index = df.columns.get_loc("proyeccion")
cols_after = df.columns[proy_index + 1:]   # columnas posteriores
df[cols_after] = df[cols_after].fillna(0)

# 5. Guardar el archivo
df.to_csv("Cre_actualizado_mensual.csv", index=False)

print("Archivo generado: Cre_actualizado_mensual.csv")
