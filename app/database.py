# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv
from sqlalchemy import text

# Cargar variables de entorno desde .env
load_dotenv()  # Busca automáticamente .env en el root del proyecto

# Leer configuración de la base de datos desde .env
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'healthpredictor')

# URL de conexión SQLAlchemy con utf8mb4 - LÍNEA CORREGIDA
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}?charset=utf8mb4"

# Crear el motor de conexión
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600  # evita errores de timeout de MySQL
)

# Crear sesión local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base para los modelos ORM
Base = declarative_base()

# Dependencia para FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def ensure_predictions_schema():
    try:
        with engine.connect() as conn:
            cols = set()
            q = text(
                """
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'predictions'
                """
            )
            res = conn.execute(q, {"db": MYSQL_DATABASE})
            for row in res:
                cols.add(row[0])

            desired = [
                ("epoc_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("arthritis_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("hepatopathy_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("parkinson_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("heart_failure_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("alzheimer_risk", "FLOAT NOT NULL DEFAULT 0"),
                ("cancer_risk", "FLOAT NOT NULL DEFAULT 0"),
            ]

            for name, type_sql in desired:
                if name not in cols:
                    try:
                        conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {name} {type_sql}"))
                    except Exception:
                        pass
            conn.commit()
    except Exception:
        pass