# backend/load_synthetic_csv.py
import pandas as pd
from sqlalchemy.orm import Session
from app.database import get_db, engine, Base
from app.models import SyntheticData
from pathlib import Path

# Ruta al CSV
CSV_PATH = Path(__file__).parent / "synthetic_data_full.csv"

# Columnas válidas según el modelo SyntheticData
allowed_cols = [
    "age","gender","bmi",
    "systolic_bp","diastolic_bp","heart_rate",
    "glucose","cholesterol_total","cholesterol_hdl","cholesterol_ldl","triglycerides",
    "smoking","alcohol_consumption","exercise_hours_per_week","stress_level","sleep_hours",
    "family_history_diabetes","family_history_hypertension","family_history_heart_disease",
    "has_diabetes","has_hypertension","has_cardiovascular_disease","has_kidney_disease",
    "has_obesity","has_dyslipidemia","has_metabolic_syndrome"
]

# Leer CSV
df = pd.read_csv(CSV_PATH)
df = df[allowed_cols]  # solo columnas que existen en el modelo

# Crear tablas si no existen
Base.metadata.create_all(bind=engine)

# Función para cargar datos
def load_data(session: Session):
    for _, row in df.iterrows():
        record = SyntheticData(
            age=int(row["age"]),
            gender=row["gender"],
            height=None,  # tu CSV no tiene altura
            weight=None,  # tu CSV no tiene peso
            bmi=float(row["bmi"]),
            systolic_bp=float(row["systolic_bp"]),
            diastolic_bp=float(row["diastolic_bp"]),
            heart_rate=float(row["heart_rate"]),
            glucose=float(row["glucose"]),
            cholesterol_total=float(row["cholesterol_total"]),
            cholesterol_hdl=float(row["cholesterol_hdl"]),
            cholesterol_ldl=float(row["cholesterol_ldl"]),
            triglycerides=float(row["triglycerides"]),
            smoking=bool(int(row["smoking"])),
            alcohol_consumption=row.get("alcohol_consumption", "none"),
            exercise_hours_per_week=float(row.get("exercise_hours_per_week", 0)),
            stress_level=int(row.get("stress_level", 5)),
            sleep_hours=float(row.get("sleep_hours", 7)),
            family_history_diabetes=bool(int(row.get("family_history_diabetes", 0))),
            family_history_hypertension=bool(int(row.get("family_history_hypertension", 0))),
            family_history_heart_disease=bool(int(row.get("family_history_heart_disease", 0))),
            has_diabetes=bool(int(row.get("has_diabetes", 0))),
            has_hypertension=bool(int(row.get("has_hypertension", 0))),
            has_cardiovascular_disease=bool(int(row.get("has_cardiovascular_disease", 0))),
            has_kidney_disease=bool(int(row.get("has_kidney_disease", 0))),
            has_obesity=bool(int(row.get("has_obesity", 0))),
            has_dyslipidemia=bool(int(row.get("has_dyslipidemia", 0))),
            has_metabolic_syndrome=bool(int(row.get("has_metabolic_syndrome", 0))),
        )
        session.add(record)

    session.commit()
    print(f"Se cargaron {len(df)} registros en SyntheticData")

# Ejecutar carga
if __name__ == "__main__":
    db = next(get_db())
    load_data(db)
