from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from config import Base

class Institution(Base):
    __tablename__ = "institution"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(150), nullable=False)
    descripcion = Column(String, nullable=True)
    fecha_creacion = Column(String, nullable=True)

    modelos = relationship("Modelos", back_populates="institution", cascade="all, delete")

    def __repr__(self):
        return f"<Institution {self.nombre}>"

class Modelos(Base):
    __tablename__ = "modelos"

    modelosid = Column(Integer, primary_key=True, index=True)
    institucionid = Column(Integer, ForeignKey("institution.id"), nullable=False, index=True)
    sucursalid = Column(Integer, index=True)
    codigo = Column(String(12), nullable=False)
    modelo = Column(String(50), nullable=True)
    ubicacion = Column(String(50), nullable=True)

    # relación con Institution
    institution = relationship("Institution", back_populates="modelos")


    def __repr__(self):
        return f"<Modelos {self.modelo}>"
