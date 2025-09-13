from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Datos de conexión
DATABASE_URL = "postgresql+psycopg2://emelchor:Emelch0r1*@localhost:5432/proyeccionweb"

# Crear motor de conexión
engine = create_engine(DATABASE_URL, echo=True)

# Base para los modelos
Base = declarative_base()

# Sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)