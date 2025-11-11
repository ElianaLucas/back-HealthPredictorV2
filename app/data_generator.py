# app/data_generator.py
import numpy as np
from sqlalchemy.orm import Session
from app.models import SyntheticData
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_data(db: Session, num_samples: int = 10000):
    """
    Generate synthetic health data for training ML models
    """
    logger.info(f"Generating {num_samples} synthetic health records...")
    np.random.seed(42)
    data_objs = []

    for _ in range(num_samples):
        age = int(np.random.randint(18, 85))
        gender = str(np.random.choice(['male', 'female']))

        height = float(np.random.normal(170 if gender == 'male' else 160, 10))
        weight = float(np.random.normal(75 if gender == 'male' else 65, 15))
        bmi = float(weight / ((height / 100) ** 2))

        systolic_bp = float(np.random.normal(120, 15))
        diastolic_bp = float(np.random.normal(80, 10))
        heart_rate = float(np.random.normal(72, 10))

        glucose = float(np.random.normal(100, 20))
        cholesterol_total = float(np.random.normal(200, 40))
        cholesterol_hdl = float(np.random.normal(50, 15))
        cholesterol_ldl = float(np.random.normal(100, 30))
        triglycerides = float(np.random.normal(150, 50))

        smoking = bool(np.random.choice([0,1], p=[0.8,0.2]))
        alcohol = str(np.random.choice(['none','moderate','high'], p=[0.6,0.3,0.1]))
        exercise = float(max(0, np.random.normal(3, 2)))
        stress = int(np.clip(np.random.randint(1, 11), 1, 10))
        sleep = float(np.clip(np.random.normal(7, 1.5), 3, 12))

        fh_diabetes = bool(np.random.choice([0,1], p=[0.85, 0.15]))
        fh_hypertension = bool(np.random.choice([0,1], p=[0.8, 0.2]))
        fh_heart = bool(np.random.choice([0,1], p=[0.9, 0.1]))

        # Generate labels heuristically (simple rules, for synthetic data)
        has_diabetes = glucose >= 126 or (bmi >= 30 and glucose >= 110)
        has_hypertension = systolic_bp >= 140 or diastolic_bp >= 90
        has_cardiovascular_disease = (has_hypertension and cholesterol_total >= 240) or (age >= 65 and smoking)
        has_kidney_disease = (glucose >= 200) or (age >= 70 and systolic_bp >= 160)
        has_obesity = bmi >= 30
        has_dyslipidemia = cholesterol_total >= 240 or triglycerides >= 200
        has_metabolic_syndrome = has_obesity and (glucose >= 100 or systolic_bp >= 130)

        obj = SyntheticData(
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            bmi=bmi,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            glucose=glucose,
            cholesterol_total=cholesterol_total,
            cholesterol_hdl=cholesterol_hdl,
            cholesterol_ldl=cholesterol_ldl,
            triglycerides=triglycerides,
            smoking=smoking,
            alcohol_consumption=alcohol,
            exercise_hours_per_week=exercise,
            stress_level=stress,
            sleep_hours=sleep,
            family_history_diabetes=fh_diabetes,
            family_history_hypertension=fh_hypertension,
            family_history_heart_disease=fh_heart,
            has_diabetes=bool(has_diabetes),
            has_hypertension=bool(has_hypertension),
            has_cardiovascular_disease=bool(has_cardiovascular_disease),
            has_kidney_disease=bool(has_kidney_disease),
            has_obesity=bool(has_obesity),
            has_dyslipidemia=bool(has_dyslipidemia),
            has_metabolic_syndrome=bool(has_metabolic_syndrome)
        )
        data_objs.append(obj)

    # bulk insert
    db.bulk_save_objects(data_objs)
    db.commit()

    logger.info(f"Successfully generated {num_samples} synthetic health records")
    return {"message": f"Generated {num_samples} synthetic records", "count": num_samples}
