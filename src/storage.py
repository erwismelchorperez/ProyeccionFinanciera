import os
import joblib


# ruta absoluta del proyecto (un nivel arriba de src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def crear_carpeta_institucion(institucion: int, sucursal: int) -> tuple:
    """
    Crea la estructura de carpetas para una institución y su sucursal.
    Retorna rutas (suc_matriz, suc_dir, plots_dir).
    """
    # carpetas bajo la raíz del proyecto
    root_dir = os.path.join(PROJECT_ROOT, f"instituciones/institucion_{institucion}")
    os.makedirs(root_dir, exist_ok=True)

    # sucursal matriz siempre existe
    suc_matriz = os.path.join(root_dir, "sucursal_0")
    os.makedirs(suc_matriz, exist_ok=True)

    suc_dir = None
    if sucursal != 0:
        suc_dir = os.path.join(root_dir, f"sucursal_{sucursal}")
        os.makedirs(suc_dir, exist_ok=True)

    plots_dir = os.path.join(PROJECT_ROOT, "plots", f"institucion_{institucion}/sucursal_{sucursal}")
    os.makedirs(plots_dir, exist_ok=True)

    return suc_matriz, suc_dir, plots_dir


def crear_carpeta_cuenta(base_dir: str, cuenta_objetivo: str) -> str:
    """
    Crea carpeta para una cuenta dentro de la institución/sucursal.
    Retorna la ruta de la carpeta creada.
    """
    cuenta_dir = os.path.join(base_dir, cuenta_objetivo)
    os.makedirs(cuenta_dir, exist_ok=True)
    return cuenta_dir


def guardar_modelo(modelo, ruta: str):
    """
    Guarda un modelo en disco usando joblib.
    """
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado en {ruta}")
