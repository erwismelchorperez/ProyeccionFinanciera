# src/db_mapeos.py
import psycopg

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "proyecciones",
    "user": "proyeccion",
    "password": "proyecc10n35",
}

def get_connection():
    return psycopg.connect(**DB_PARAMS)

def obtener_mapeo_codigos(templateid: int) -> dict:
    print("→ entrando a obtener_mapeo_codigos", flush=True)
    conn = get_connection()
    print("→ conexión creada", flush=True)
    cur = conn.cursor()
    print("→ cursor creado", flush=True)
    cur.execute("""
        SELECT codigo, cuentaid
        FROM cuenta_contable
        WHERE templateid = %s
    """, (templateid,))
    print("→ consulta ejecutada", flush=True)
    mapping = {str(codigo): cuentaid for codigo, cuentaid in cur.fetchall()}
    print("→ fetchall hecho", flush=True)
    cur.close()
    conn.close()
    print("→ conexión cerrada", flush=True)
    return mapping
