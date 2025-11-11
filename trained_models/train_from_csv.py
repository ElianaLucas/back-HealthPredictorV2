# backend/train_from_csv.py
import pandas as pd
from pathlib import Path
from ml_models import FEATURE_COLUMNS, DISEASES, _prepare_df, train_disease_model
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ruta de tu CSV
CSV_PATH = Path(__file__).parent.joinpath("synthetic_data_full.csv")

if not CSV_PATH.exists():
    logger.error(f"CSV not found: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
logger.info(f"Loaded CSV with {len(df)} rows")

# Carpeta para guardar modelos (ajustada al backend/trained_models)
MODELS_DIR = Path(__file__).parent.joinpath("trained_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Models will be saved in {MODELS_DIR}")

# Preparar features
X = _prepare_df(df)

# Entrenar cada enfermedad
results = {}
for disease_name, label_col in DISEASES:
    if label_col not in df.columns:
        logger.warning(f"Label column not found: {label_col} - skipping {disease_name}")
        results[disease_name] = {"skipped": True, "reason": "label_not_found"}
        continue
    y = df[label_col].astype(int)
    res = train_disease_model(X, y, disease_name)
    results[disease_name] = res

# Mostrar resumen
for disease, r in results.items():
    if r.get("skipped"):
        logger.info(f"{disease}: skipped ({r.get('reason')})")
    else:
        logger.info(f"{disease}: accuracy={r['accuracy']:.2f}, auc={r['auc']:.2f}, saved at {r['model_path']}")

logger.info("All models trained and saved!")
