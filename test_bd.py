import psycopg

DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "proyecciones",
    "user": "proyeccion",
    "password": "proyecc10n35"
}

print("Conectando...")
conn = psycopg.connect(**DB_PARAMS)
print("Conexión OK")
cur = conn.cursor()
cur.execute("SELECT codigo, cuentaid FROM cuenta_contable WHERE templateid=17;")
print("Resultado:", cur.fetchone())
cur.close()
conn.close()
print("Todo bien.")
