# backend/train_all_diseases.py
from sqlalchemy.orm import Session
from app.database import get_db
from app.ml_models import train_all_models

if __name__ == "__main__":
    db: Session = next(get_db())
    results = train_all_models(db)
    print("Resultados del entrenamiento:")
    for disease, metrics in results.get("results", {}).items():
        print(disease, metrics)
