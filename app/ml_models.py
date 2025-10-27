# app/ml_models.py
import os
from pathlib import Path
import joblib
import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sqlalchemy.orm import Session
from models import SyntheticData

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.joinpath("trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gender_male'] = (df['gender'] == 'male').astype(int)
    alcohol_map = {'none': 0, 'moderate': 1, 'high': 2}
    df['alcohol_numeric'] = df['alcohol_consumption'].map(alcohol_map).fillna(0).astype(int)
    bool_columns = ['smoking', 'family_history_diabetes', 'family_history_hypertension', 'family_history_heart_disease']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    feature_columns = [
        'age', 'gender_male', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'glucose', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl', 'triglycerides',
        'smoking', 'alcohol_numeric', 'exercise_hours_per_week', 'stress_level', 'sleep_hours',
        'family_history_diabetes', 'family_history_hypertension', 'family_history_heart_disease'
    ]
    return df[feature_columns]

def train_disease_model(X, y, disease_name, test_size=0.2):
    logger.info(f"Training model for {disease_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"{disease_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    model_path = MODELS_DIR.joinpath(f"{disease_name}_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    return {"disease": disease_name, "accuracy": float(accuracy), "auc": float(auc), "model_path": str(model_path)}

def train_all_models(db: Session):
    logger.info("Starting model training for all diseases...")
    data = db.query(SyntheticData).all()
    if len(data) == 0:
        raise ValueError("No synthetic data found. Please generate data first.")
    df = pd.DataFrame([{
        'age': d.age,
        'gender': d.gender,
        'bmi': d.bmi,
        'systolic_bp': d.systolic_bp,
        'diastolic_bp': d.diastolic_bp,
        'heart_rate': d.heart_rate,
        'glucose': d.glucose,
        'cholesterol_total': d.cholesterol_total,
        'cholesterol_hdl': d.cholesterol_hdl,
        'cholesterol_ldl': d.cholesterol_ldl,
        'triglycerides': d.triglycerides,
        'smoking': d.smoking,
        'alcohol_consumption': d.alcohol_consumption,
        'exercise_hours_per_week': d.exercise_hours_per_week,
        'stress_level': d.stress_level,
        'sleep_hours': d.sleep_hours,
        'family_history_diabetes': d.family_history_diabetes,
        'family_history_hypertension': d.family_history_hypertension,
        'family_history_heart_disease': d.family_history_heart_disease,
        'has_diabetes': d.has_diabetes,
        'has_hypertension': d.has_hypertension,
        'has_cardiovascular_disease': d.has_cardiovascular_disease,
        'has_kidney_disease': d.has_kidney_disease,
        'has_obesity': d.has_obesity,
        'has_dyslipidemia': d.has_dyslipidemia,
        'has_metabolic_syndrome': d.has_metabolic_syndrome
    } for d in data])
    X = prepare_features(df)
    diseases = [
        ('diabetes', 'has_diabetes'),
        ('hypertension', 'has_hypertension'),
        ('cardiovascular', 'has_cardiovascular_disease'),
        ('kidney_disease', 'has_kidney_disease'),
        ('obesity', 'has_obesity'),
        ('dyslipidemia', 'has_dyslipidemia'),
        ('metabolic_syndrome', 'has_metabolic_syndrome')
    ]
    results = []
    for disease_name, label_column in diseases:
        y = df[label_column].astype(int)
        result = train_disease_model(X, y, disease_name)
        results.append(result)
    logger.info("All models trained successfully")
    return results

def load_models():
    models = {}
    disease_names = ['diabetes', 'hypertension', 'cardiovascular', 'kidney_disease', 'obesity', 'dyslipidemia', 'metabolic_syndrome']
    for disease in disease_names:
        model_path = MODELS_DIR.joinpath(f"{disease}_model.pkl")
        if model_path.exists():
            models[disease] = joblib.load(model_path)
        else:
            logger.warning(f"Model not found: {model_path}")
    return models

def predict_risks(patient_data: dict):
    models = load_models()
    if not models:
        raise ValueError("No trained models found. Please train models first.")
    import pandas as pd
    df = pd.DataFrame([patient_data])
    X = prepare_features(df)
    predictions = {}
    for disease, model in models.items():
        proba = model.predict_proba(X)[0, 1]
        risk_score = float(proba * 100)
        predictions[f"{disease}_risk"] = round(risk_score, 2)
    return predictions
