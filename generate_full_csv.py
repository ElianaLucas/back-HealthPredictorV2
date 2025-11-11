# backend/generate_full_csv.py
import pandas as pd
import numpy as np
from pathlib import Path

# Número de filas a generar
N = 2000
CSV_PATH = Path(__file__).parent.joinpath("synthetic_data_full.csv")

np.random.seed(42)

# ----------------------------
# Funciones auxiliares
# ----------------------------

# Grupos etarios
def generate_age():
    age_groups = [(2,12), (13,25), (26,64), (65,90)]
    weights = [0.15, 0.2, 0.5, 0.15]
    group = np.random.choice(range(4), p=weights)
    return np.random.randint(age_groups[group][0], age_groups[group][1]+1)

def generate_gender():
    return np.random.choice(["male", "female"])

def generate_bool():
    return np.random.choice([0,1])

def generate_alcohol():
    return np.random.choice(["none","low","moderate","high"])

def random_range(val_range):
    return np.random.uniform(val_range[0], val_range[1])

# ----------------------------
# Generación de datos
# ----------------------------

data = []

for _ in range(N):
    age = generate_age()
    gender = generate_gender()
    
    # Rangos por edad para valores clínicos básicos
    if age <= 12:
        chol_tot = random_range((120,170))
        chol_hdl = random_range((40,65))
        chol_ldl = random_range((60,110))
        glucose = random_range((70,100))
        systolic_bp = random_range((90,110))
        diastolic_bp = random_range((50,70))
    elif age <= 25:
        chol_tot = random_range((125,200))
        chol_hdl = random_range((40,70))
        chol_ldl = random_range((60,110))
        glucose = random_range((70,100))
        systolic_bp = random_range((110,120))
        diastolic_bp = random_range((70,80))
    elif age <= 64:
        chol_tot = random_range((125,200))
        chol_hdl = random_range((40,70))
        chol_ldl = random_range((60,130))
        glucose = random_range((70,100))
        systolic_bp = random_range((110,130))
        diastolic_bp = random_range((70,85))
    else:
        chol_tot = random_range((150,240))
        chol_hdl = random_range((40,70))
        chol_ldl = random_range((70,130))
        glucose = random_range((80,130))
        systolic_bp = random_range((120,140))
        diastolic_bp = random_range((70,90))
    
    # Otros valores generales
    bmi = random_range((15,35))
    heart_rate = random_range((60,100))
    triglycerides = random_range((50,200))
    smoking = generate_bool()
    alcohol = generate_alcohol()
    exercise = random_range((0,10))
    stress = random_range((1,10))
    sleep = random_range((4,9))
    fam_diab = generate_bool()
    fam_hyp = generate_bool()
    fam_heart = generate_bool()
    
    # Labels 0/1 (aleatorio por ahora)
    labels = [generate_bool() for _ in range(19)]  # <-- CORREGIDO: 19 labels

    row = [
        age, gender, bmi, systolic_bp, diastolic_bp, heart_rate,
        glucose, chol_tot, chol_hdl, chol_ldl, triglycerides,
        smoking, alcohol, exercise, stress, sleep,
        fam_diab, fam_hyp, fam_heart
    ] + labels

    data.append(row)

# ----------------------------
# Columnas
# ----------------------------

columns = [
    "age","gender","bmi","systolic_bp","diastolic_bp","heart_rate",
    "glucose","cholesterol_total","cholesterol_hdl","cholesterol_ldl",
    "triglycerides","smoking","alcohol_consumption","exercise_hours_per_week",
    "stress_level","sleep_hours","family_history_diabetes",
    "family_history_hypertension","family_history_heart_disease",
    "has_diabetes","has_hypertension","has_cardiovascular_disease","has_kidney_disease",
    "has_obesity","has_dyslipidemia","has_metabolic_syndrome",
    "has_hypothyroidism","has_anemia_chronic","has_osteoporosis","has_epoc",
    "has_arthritis","has_liver_disease","has_parkinson","has_heart_failure",
    "has_multiple_sclerosis","has_cirrhosis","has_alzheimer","has_cancer"
]

# ----------------------------
# Crear DataFrame y guardar CSV
# ----------------------------

df = pd.DataFrame(data, columns=columns)
df.to_csv(CSV_PATH, index=False)
print(f"CSV generado en {CSV_PATH} con {N} filas")
