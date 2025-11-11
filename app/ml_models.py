# app/ml_models.py
import os
from pathlib import Path
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from .models import SyntheticData

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Carpeta para guardar modelos
MODELS_DIR = Path(__file__).parent.parent.joinpath("trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Columnas/features que usaremos
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "glucose",
    "cholesterol_total",
    "cholesterol_hdl",
    "cholesterol_ldl",
    "triglycerides",
    "smoking",
    "exercise_hours_per_week",
    "stress_level",
    "sleep_hours",
    "family_history_diabetes",
    "family_history_hypertension",
    "family_history_heart_disease",
    "gender_male",
    "alcohol_numeric",
    # Nuevos biomarcadores para enfermedades crónicas
    "fev1",
    "pef",
    "ige_total",
    "fr",
    "pcr",
    "vsg",
    "alt",
    "ast",
    "bilirubin_total",
    "albumin",
    "abdominal_circumference",
    "updrs",
    "dopamine",
    "walk_time",
    "fev_heart",
    "bnp",
    "ntprobnp",
    "bands_oligo",
    "igg_index",
    "edss",
    "child_pugh",
    "inr",
    "prot_time",
    "mmse",
    "moca",
    "tau",
    "cea",
    "ca19_9",
    "psa",
    "hemoglobin"
]

# Enfermedades y columnas objetivo (targets)
DISEASES: List[tuple] = [
    ("diabetes", "has_diabetes"),
    ("hypertension", "has_hypertension"),
    ("obesity", "has_obesity"),
    ("cardiovascular", "has_cardiovascular_disease"),
    ("kidney_disease", "has_kidney_disease"),
    ("dyslipidemia", "has_dyslipidemia"),
    ("metabolic_syndrome", "has_metabolic_syndrome"),
    ("epoc", "has_epoc"),
    ("arthritis", "has_arthritis"),
    ("hepatopathy", "has_hepatopathy"),
    ("parkinson", "has_parkinson"),
    ("heart_failure", "has_heart_failure"),
    ("alzheimer", "has_alzheimer"),
    ("cancer", "has_cancer"),
]

def _model_path(disease_name: str) -> Path:
    return MODELS_DIR.joinpath(f"{disease_name}_xgb.joblib")

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gender -> gender_male
    def _is_male(g):
        if pd.isna(g):
            return 0
        gstr = str(g).lower()
        return 1 if gstr in ("male", "m", "masculino", "h", "hombre") else 0

    df["gender_male"] = df.get("gender", "").apply(_is_male) if "gender" in df.columns else 0

    # Alcohol mapping
    alcohol_map = {"none": 0, "moderate": 1, "high": 2, "low": 0}
    if "alcohol_consumption" in df.columns:
        df["alcohol_numeric"] = df["alcohol_consumption"].map(
            lambda v: alcohol_map.get(str(v).lower(), 0)
        )
    else:
        df["alcohol_numeric"] = 0

    # Boolean columns
    bool_cols = [
        "smoking",
        "family_history_diabetes",
        "family_history_hypertension",
        "family_history_heart_disease",
        "bands_oligo"  # boolean para EM
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
        else:
            df[c] = 0

    # Ensure all feature columns present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)
    return df[FEATURE_COLUMNS].astype(float)

def _load_synthetic_dataframe(db: Session) -> pd.DataFrame:
    from .models import SyntheticData
    rows = []
    for d in db.query(SyntheticData).all():
        row = {f: getattr(d, f, 0) for f in FEATURE_COLUMNS}
        # labels
        for disease_name, label in DISEASES:
            row[label] = int(getattr(d, label, 0))
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def train_disease_model(X: pd.DataFrame, y: pd.Series, disease_name: str, test_size: float = 0.2) -> Dict[str, Any]:
    logger.info(f"Training {disease_name} model - samples: {len(X)}")
    if y.nunique() < 2 or len(y) < 10:
        logger.warning(f"Not enough variation to train {disease_name}")
        return {"disease": disease_name, "skipped": True, "reason": "insufficient_data"}
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = float("nan")
    path = _model_path(disease_name)
    joblib.dump(model, path)
    logger.info(f"Saved model for {disease_name} -> {path}")
    return {"disease": disease_name, "accuracy": float(acc), "auc": auc, "model_path": str(path)}

def train_all_models(db: Session) -> Dict[str, Any]:
    df = _load_synthetic_dataframe(db)
    if df.empty:
        return {"status": "no_data"}
    X = _prepare_df(df)
    results = {}
    for disease_name, label_col in DISEASES:
        if label_col not in df.columns:
            results[disease_name] = {"skipped": True, "reason": "label_not_found"}
            continue
        y = df[label_col].astype(int)
        results[disease_name] = train_disease_model(X, y, disease_name)
    return {"status": "done", "results": results}

def load_models() -> Dict[str, Any]:
    models = {}
    for disease_name, _ in DISEASES:
        path = _model_path(disease_name)
        if path.exists():
            try:
                models[disease_name] = joblib.load(path)
                logger.info(f"Loaded model: {path}")
            except Exception as e:
                logger.error(f"Failed loading model {path}: {e}")
    return models

def predict_risks(patient_data: Dict[str, Any]) -> Dict[str, float]:
    models = load_models()
    df = pd.DataFrame([patient_data])
    X = _prepare_df(df)
    if not models:
        # fallback heuristics simple
        return {f"{d}_risk": 0.0 for d, _ in DISEASES}
    out = {}
    for disease_name, _ in DISEASES:
        model = models.get(disease_name)
        if model is None:
            out[f"{disease_name}_risk"] = 0.0
            continue
        try:
            proba = model.predict_proba(X)[0, 1]
            out[f"{disease_name}_risk"] = float(round(proba * 100, 2))
        except Exception as e:
            logger.error(f"Prediction error for {disease_name}: {e}")
            out[f"{disease_name}_risk"] = 0.0
    return out
