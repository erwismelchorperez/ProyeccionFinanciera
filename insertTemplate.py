import pandas as pd
import psycopg

# --- Configuración ---
CSV_PATH = "./dataset/Estados_FinancierosGaby_proyeccion.csv"   # ruta al CSV
DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "proyecciones",
    "user": "proyeccion",     # tu usuario
    "password": "proyecc10n35"
}

# --- Leer CSV ---
df = pd.read_csv(CSV_PATH)

# Renombrar columnas para facilitar
df = df.rename(columns={
    "BALANCE GENERAL": "nombre",
    "NIVEL": "nivel",
    "Codigo": "codigo",
    "proyeccion": "proyeccion"
})

# Filtrar solo filas donde 'nivel' sea numérico
df = df[pd.to_numeric(df["nivel"], errors="coerce").notna()]
df["nivel"] = df["nivel"].astype(int)

# --- Conexión ---
conn = psycopg.connect(**DB_PARAMS)
cur = conn.cursor()

TEMPLATE_ID = 19      # <<--- templateid que ya existe
TIPO_ID = 1           # tipocuentaid = 1

# --- Insertar fila por fila ---
for _, row in df.iterrows():
    proy = str(row["proyeccion"]).strip().upper()[:2]  # char(2)
    
    cur.execute("""
        INSERT INTO cuenta_contable (templateid, nivel, tipoid, codigo, nombre, proyeccion)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        TEMPLATE_ID,              # templateid = 19
        int(row["nivel"]),        # nivel del CSV
        TIPO_ID,                  # tipoid fijo (debe existir en tipocuenta)
        int(row["codigo"]),       # codigo como texto (varchar(50))
        str(row["nombre"]),       # nombre
        proy                      # proyeccion (2 caracteres)
    ))

conn.commit()
cur.close()
conn.close()

print("Inserciones completadas")
