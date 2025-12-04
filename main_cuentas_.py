# main_cuenta.py

import psycopg
from psycopg.rows import dict_row
# NO importes tensorflow aquí arriba

from src.insertar_modelos import obtener_mapeo_codigos, insertar_modelo, get_connection
from src.storage import crear_carpeta_cuenta,crear_carpeta_institucion,guardar_modelo
# y demás imports normales (sin TF)


def run(institucion: int, sucursal: int, templateid: int):
    suc_matriz, suc_dir, plots_dir = crear_carpeta_institucion(institucion, sucursal)
    print("INICIO")
    codigo_to_id = obtener_mapeo_codigos(templateid)
    print("→ mapeo listo, total códigos:", len(codigo_to_id))

    # SOLO DESPUÉS DE TENER EL MAPE0, importa TensorFlow
    import tensorflow as tf
    from tensorflow import keras

    print("→ TensorFlow importado, versión:", tf.__version__)
    # aquí ya sigues con tu lógica de modelos, entrenamiento, etc.
    # usar codigo_to_id para lo que necesites


if __name__ == "__main__":
    import sys
    institucion = int(sys.argv[1])
    sucursal = int(sys.argv[2])
    templateid = int(sys.argv[3])
    run(institucion, sucursal, templateid)
