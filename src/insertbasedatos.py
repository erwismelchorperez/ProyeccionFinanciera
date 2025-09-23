from config import Base, engine, SessionLocal
from models import Institution, Modelos
"""
    Crear tablas en la base de datos, esto lo hace porque se crean desde el archivo models.py, 
    así podemos crear las tablas desde aca y no necesariamente desde el gestor de base de datos
"""
Base.metadata.create_all(bind=engine)
# Iniciar sesión
session = SessionLocal()


try:
    # Operaciones con la BD
    new_item = Modelos(institucionid=1, sucursalid=1, codigo="101", modelo="Test", ubicacion="./../entidades/Linear_101..pkl")
    session.add(new_item)
    session.commit()
    session.refresh(new_item)
finally:
    session.close()

