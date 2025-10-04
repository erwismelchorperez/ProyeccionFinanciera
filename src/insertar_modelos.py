import psycopg
from psycopg.rows import dict_row  # opcional, filas como dict
import json

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "-",
    "user": "antonio",
    "password": "-"
}

def get_connection():
    """Crea y retorna una conexión a la BD."""
    return psycopg.connect(**DB_PARAMS)

def insertar_modelo(cuentaid: int, modelo: str, ubicacion: str, variables: dict = None):
    """
    Inserta un modelo en la tabla modelos.
    - cuentaid: id de la cuenta contable
    - modelo: nombre del modelo (ej. Lasso_3_Disponibilidades.pkl)
    - ubicacion: ruta en disco
    - variables: diccionario de métricas (se guarda como JSONB si existe)
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            INSERT INTO modelos (cuentaid, modelo, ubicacion)
            VALUES (%s, %s, %s)
        """, (
            cuentaid,
            modelo,
            ubicacion,
        ))
        conn.commit()
        print(f"Insertado en BD: cuentaid={cuentaid}, modelo={modelo}")
    except Exception as e:
        conn.rollback()
        print(f"⚠️ Error al insertar modelo {modelo}: {e}")
    finally:
        cur.close()
        conn.close()

def obtener_mapeo_codigos(templateid: int) -> dict:
    """
    Devuelve un diccionario {codigo: cuentaid} para un templateid dado.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT codigo, cuentaid
        FROM cuenta_contable
        WHERE templateid = %s
    """, (templateid,))
    mapping = {str(codigo): cuentaid for codigo, cuentaid in cur.fetchall()}
    cur.close()
    conn.close()
    return mapping
