# app/server.py
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from database import engine, get_db, Base
from models import User, PatientRecord, Prediction
from auth import get_password_hash, verify_password, create_access_token, get_current_user
from data_generator import generate_synthetic_data
from ml_models import train_all_models, predict_risks
from pdf_generator import generate_pdf_report
from pydantic import BaseModel, EmailStr, constr
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent  # sube un nivel
load_dotenv(ROOT_DIR / '.env')

# Create tables
Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Risk Prediction System")
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Pydantic schemas (responses/requests) =====
class UserRegister(BaseModel):
    email: EmailStr
    password: constr(min_length=4) 
    full_name: str
    role: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class PatientRecordCreate(BaseModel):
    age: int = Field(ge=0, le=150)
    gender: str
    height: float = Field(gt=0)
    weight: float = Field(gt=0)
    systolic_bp: float = Field(gt=0)
    diastolic_bp: float = Field(gt=0)
    heart_rate: float = Field(gt=0)
    glucose: float = Field(gt=0)
    cholesterol_total: float = Field(gt=0)
    cholesterol_hdl: float = Field(gt=0)
    cholesterol_ldl: float = Field(gt=0)
    triglycerides: float = Field(gt=0)
    smoking: bool = False
    alcohol_consumption: str = "none"
    exercise_hours_per_week: float = Field(ge=0)
    stress_level: int = Field(ge=1, le=10)
    sleep_hours: float = Field(ge=0, le=24)
    family_history_diabetes: bool = False
    family_history_hypertension: bool = False
    family_history_heart_disease: bool = False
    current_medications: Optional[str] = None

class PatientRecordResponse(BaseModel):
    id: str
    age: int
    gender: str
    height: float
    weight: float
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    glucose: float
    cholesterol_total: float
    cholesterol_hdl: float
    cholesterol_ldl: float
    triglycerides: float
    smoking: bool
    alcohol_consumption: str
    exercise_hours_per_week: float
    stress_level: int
    sleep_hours: float
    family_history_diabetes: bool
    family_history_hypertension: bool
    family_history_heart_disease: bool
    current_medications: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    id: str
    patient_record_id: str
    diabetes_risk: float
    hypertension_risk: float
    cardiovascular_risk: float
    kidney_disease_risk: float
    obesity_risk: float
    dyslipidemia_risk: float
    metabolic_syndrome_risk: float
    overall_risk: str
    recommendations: Optional[dict]
    created_at: datetime
    class Config:
        from_attributes = True

# ===== Helper functions =====
def calculate_bmi(weight: float, height: float) -> float:
    height_m = height / 100
    return weight / (height_m ** 2)

def calculate_overall_risk(risks: dict) -> str:
    risk_values = [
        risks.get('diabetes_risk', 0),
        risks.get('hypertension_risk', 0),
        risks.get('cardiovascular_risk', 0),
        risks.get('kidney_disease_risk', 0),
        risks.get('obesity_risk', 0),
        risks.get('dyslipidemia_risk', 0),
        risks.get('metabolic_syndrome_risk', 0)
    ]
    max_risk = max(risk_values)
    avg_risk = sum(risk_values) / len(risk_values)
    if max_risk >= 80 or avg_risk >= 60:
        return "very_high"
    elif max_risk >= 60 or avg_risk >= 40:
        return "high"
    elif max_risk >= 40 or avg_risk >= 20:
        return "medium"
    else:
        return "low"

def generate_recommendations(patient_data: dict, risks: dict) -> dict:
    recommendations = {"lifestyle": [], "medical": [], "monitoring": []}
    if patient_data['bmi'] >= 30:
        recommendations["lifestyle"].append("Reducir peso mediante dieta equilibrada y ejercicio regular")
    elif patient_data['bmi'] >= 25:
        recommendations["lifestyle"].append("Mantener peso saludable con dieta balanceada")
    if patient_data['exercise_hours_per_week'] < 2.5:
        recommendations["lifestyle"].append("Aumentar actividad física a al menos 150 minutos por semana")
    if patient_data['smoking']:
        recommendations["lifestyle"].append("Dejar de fumar - considerar programas de cesación tabáquica")
    if patient_data['systolic_bp'] >= 140 or patient_data['diastolic_bp'] >= 90:
        recommendations["medical"].append("Consultar con médico sobre presión arterial elevada")
        recommendations["monitoring"].append("Monitorear presión arterial regularmente")
    if patient_data['glucose'] >= 126:
        recommendations["medical"].append("Consultar con endocrinólogo - posible diabetes")
        recommendations["monitoring"].append("Monitorear niveles de glucosa en sangre")
    elif patient_data['glucose'] >= 100:
        recommendations["medical"].append("Evaluación de prediabetes")
    if patient_data['cholesterol_total'] >= 240:
        recommendations["medical"].append("Consultar con médico sobre colesterol alto")
        recommendations["lifestyle"].append("Dieta baja en grasas saturadas y trans")
    if patient_data['stress_level'] >= 7:
        recommendations["lifestyle"].append("Técnicas de manejo del estrés: meditación, yoga, terapia")
    if patient_data['sleep_hours'] < 6:
        recommendations["lifestyle"].append("Mejorar higiene del sueño - objetivo: 7-9 horas por noche")
    if risks.get('diabetes_risk', 0) >= 60:
        recommendations["medical"].append("Evaluación completa de diabetes - hemoglobina A1C")
    if risks.get('cardiovascular_risk', 0) >= 60:
        recommendations["medical"].append("Evaluación cardiovascular completa - ECG, ecocardiograma")
    if risks.get('kidney_disease_risk', 0) >= 60:
        recommendations["medical"].append("Evaluación de función renal - creatinina, tasa de filtración")
    recommendations["monitoring"].append("Chequeo médico anual completo")
    recommendations["monitoring"].append("Análisis de sangre cada 6-12 meses")
    return recommendations

# ===== Routes =====
@api_router.get("/")
async def root():
    return {"message": "Health Risk Prediction API", "version": "1.0"}

@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=user_data.role
        )
        logger.info(f"Creando usuario: {new_user.email}")
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        access_token = create_access_token(data={"sub": new_user.id})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": new_user.id,
                "email": new_user.email,
                "full_name": new_user.full_name,
                "role": new_user.role
            }
        }
    except Exception as e:
        logger.error(f"Error en registro: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer", "user": {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "email": current_user.email, "full_name": current_user.full_name, "role": current_user.role}

@api_router.post("/patient-records", response_model=PatientRecordResponse)
async def create_patient_record(record_data: PatientRecordCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bmi = calculate_bmi(record_data.weight, record_data.height)
    record_dict = record_data.model_dump() if hasattr(record_data, "model_dump") else record_data.dict()
    patient_record = PatientRecord(user_id=current_user.id, bmi=bmi, **record_dict)
    db.add(patient_record); db.commit(); db.refresh(patient_record)
    return patient_record

@api_router.get("/patient-records", response_model=List[PatientRecordResponse])
async def get_patient_records(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = db.query(PatientRecord).filter(PatientRecord.user_id == current_user.id).order_by(PatientRecord.created_at.desc()).all()
    return records

@api_router.get("/patient-records/{record_id}", response_model=PatientRecordResponse)
async def get_patient_record(record_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    record = db.query(PatientRecord).filter(PatientRecord.id == record_id, PatientRecord.user_id == current_user.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

@api_router.post("/predictions/{record_id}", response_model=PredictionResponse)
async def create_prediction(record_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient_record = db.query(PatientRecord).filter(PatientRecord.id == record_id, PatientRecord.user_id == current_user.id).first()
    if not patient_record:
        raise HTTPException(status_code=404, detail="Patient record not found")

    patient_data = {
        'age': patient_record.age,
        'gender': patient_record.gender,
        'bmi': patient_record.bmi,
        'systolic_bp': patient_record.systolic_bp,
        'diastolic_bp': patient_record.diastolic_bp,
        'heart_rate': patient_record.heart_rate,
        'glucose': patient_record.glucose,
        'cholesterol_total': patient_record.cholesterol_total,
        'cholesterol_hdl': patient_record.cholesterol_hdl,
        'cholesterol_ldl': patient_record.cholesterol_ldl,
        'triglycerides': patient_record.triglycerides,
        'smoking': patient_record.smoking,
        'alcohol_consumption': patient_record.alcohol_consumption,
        'exercise_hours_per_week': patient_record.exercise_hours_per_week,
        'stress_level': patient_record.stress_level,
        'sleep_hours': patient_record.sleep_hours,
        'family_history_diabetes': patient_record.family_history_diabetes,
        'family_history_hypertension': patient_record.family_history_hypertension,
        'family_history_heart_disease': patient_record.family_history_heart_disease
    }

    try:
        risks = predict_risks(patient_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    overall_risk = calculate_overall_risk(risks)
    recommendations = generate_recommendations(patient_data, risks)

    prediction = Prediction(
        user_id=current_user.id,
        patient_record_id=record_id,
        diabetes_risk=risks.get('diabetes_risk', 0.0),
        hypertension_risk=risks.get('hypertension_risk', 0.0),
        cardiovascular_risk=risks.get('cardiovascular_risk', 0.0),
        kidney_disease_risk=risks.get('kidney_disease_risk', 0.0),
        obesity_risk=risks.get('obesity_risk', 0.0),
        dyslipidemia_risk=risks.get('dyslipidemia_risk', 0.0),
        metabolic_syndrome_risk=risks.get('metabolic_syndrome_risk', 0.0),
        overall_risk=overall_risk,
        recommendations=recommendations
    )

    db.add(prediction); db.commit(); db.refresh(prediction)
    return prediction

@api_router.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.created_at.desc()).all()
    return predictions

@api_router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@api_router.post("/data/generate")
async def generate_data(num_samples: int = 10000, db: Session = Depends(get_db)):
    try:
        result = generate_synthetic_data(db, num_samples)
        return result
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/models/train")
async def train_models(db: Session = Depends(get_db)):
    try:
        results = train_all_models(db)
        return {"message": "All models trained successfully", "results": results}
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/pdf/{prediction_id}")
async def get_pdf_report(prediction_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    patient_record = db.query(PatientRecord).filter(PatientRecord.id == prediction.patient_record_id).first()
    user = current_user
    pdf_path = generate_pdf_report(user, patient_record, prediction)
    from fastapi.responses import FileResponse
    return FileResponse(path=pdf_path, filename=os.path.basename(pdf_path), media_type='application/pdf')

# include router and shutdown
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
