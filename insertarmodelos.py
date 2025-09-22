import os
import psycopg2
from pathlib import Path

# --- Configuración de conexión ---
DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "-",
    "user": "-",          # cambia por tu usuario
    "password": "-"   # cambia por tu contraseña
}

# --- Ruta raíz donde están las carpetas ---
#ROOT_PATH = "/home/user/Code/ProyeccionFinanciera/instituciones/institucion_1/sucursal_0"
ROOT_PATH = "/root/proyecciones/ProyeccionFinanciera/instituciones/institucion_1/sucursal_0"
# --- Conectar a la base ---
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

# --- Obtener mapeo {codigo: cuentaid} de cuenta_contable con templateid=3 ---
cur.execute("""
    SELECT codigo, cuentaid
    FROM cuenta_contable
    WHERE templateid = 3
""")
codigo_to_id = {str(codigo): cuentaid for codigo, cuentaid in cur.fetchall()}

print("Mapeo codigo → cuentaid:", codigo_to_id)

# --- Recorrer carpetas en ROOT_PATH ---
root = Path(ROOT_PATH)

for carpeta in root.iterdir():
    if carpeta.is_dir():
        codigo = carpeta.name  # nombre de la carpeta = codigo
        if codigo not in codigo_to_id:
            print(f"Código {codigo} no encontrado en cuenta_contable, se omite.")
            continue

        cuentaid = codigo_to_id[codigo]

        # Buscar archivos dentro de la carpeta
        for archivo in carpeta.iterdir():
            if archivo.is_file():
                modelo_nombre = archivo.name
                ubicacion = str(archivo.resolve())

                print(f"Inserto: cuentaid={cuentaid}, modelo={modelo_nombre}, ubicacion={ubicacion}")

                cur.execute("""
                    INSERT INTO modelos (cuentaid, modelo, ubicacion)
                    VALUES (%s, %s, %s)
                """, (cuentaid, modelo_nombre, ubicacion))

# --- Confirmar cambios ---
conn.commit()

cur.close()
conn.close()

print("Inserciones completadas en la tabla modelos")
