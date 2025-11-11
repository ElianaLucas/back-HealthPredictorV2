# app/server.py
#venv\Scripts\uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc
from pydantic import BaseModel, EmailStr, Field, constr
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4

# Imports absolutos del proyecto (corregidos)
from app.models import User, PatientRecord, Prediction
from app.database import Base, engine, get_db, ensure_predictions_schema
from app.auth import get_password_hash, verify_password, create_access_token, get_current_user
from app.data_generator import generate_synthetic_data
from app.ml_models import train_all_models, predict_risks
from app.pdf_generator import generate_pdf_report

# Configuración básica
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

Base.metadata.create_all(bind=engine)
try:
    ensure_predictions_schema()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Health Risk Prediction System")
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
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

class PredictionWithUserResponse(BaseModel):
    id: str
    user_id: str
    patient_record_id: str
    overall_risk: str
    created_at: datetime
    patient_name: Optional[str] = None
    patient_email: Optional[EmailStr] = None
    role: Optional[str] = None

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role_id: Optional[int] = None
    is_active: Optional[bool] = None

class PatientRecordWithUser(BaseModel):
    id: str
    user_id: str
    age: int
    gender: str
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
    created_at: datetime
    user_number: Optional[int] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None

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
    epoc_risk: float
    arthritis_risk: float
    hepatopathy_risk: float
    parkinson_risk: float
    heart_failure_risk: float
    alzheimer_risk: float
    cancer_risk: float
    overall_risk: str
    recommendations: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True

# ===== Helper functions =====
def calculate_bmi(weight: float, height: float) -> float:
    height_m = height / 100.0
    return weight / (height_m ** 2) if height_m > 0 else 0.0

def calculate_overall_risk(risks: dict) -> str:
    risk_values = [
        risks.get("diabetes_risk", 0),
        risks.get("hypertension_risk", 0),
        risks.get("cardiovascular_risk", 0),
        risks.get("kidney_disease_risk", 0),
        risks.get("obesity_risk", 0),
        risks.get("dyslipidemia_risk", 0),
        risks.get("metabolic_syndrome_risk", 0),
    ]
    # defensiva: si no hay valores
    if not risk_values:
        return "low"
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
    try:
        bmi = float(patient_data.get("bmi", 0))
    except Exception:
        bmi = 0.0

    if bmi >= 30:
        recommendations["lifestyle"].append("Reducir peso mediante dieta equilibrada y ejercicio regular")
    elif bmi >= 25:
        recommendations["lifestyle"].append("Mantener peso saludable con dieta balanceada")

    if patient_data.get("exercise_hours_per_week", 0) < 2.5:
        recommendations["lifestyle"].append("Aumentar actividad física a al menos 150 minutos por semana")

    if patient_data.get("smoking"):
        recommendations["lifestyle"].append("Dejar de fumar - considerar programas de cesación tabáquica")

    if patient_data.get("systolic_bp", 0) >= 140 or patient_data.get("diastolic_bp", 0) >= 90:
        recommendations["medical"].append("Consultar con médico sobre presión arterial elevada")
        recommendations["monitoring"].append("Monitorear presión arterial regularmente")

    glucose = patient_data.get("glucose", 0)
    if glucose >= 126:
        recommendations["medical"].append("Consultar con endocrinólogo - posible diabetes")
        recommendations["monitoring"].append("Monitorear niveles de glucosa en sangre")
    elif glucose >= 100:
        recommendations["medical"].append("Evaluación de prediabetes")

    if patient_data.get("cholesterol_total", 0) >= 240:
        recommendations["medical"].append("Consultar con médico sobre colesterol alto")
        recommendations["lifestyle"].append("Dieta baja en grasas saturadas y trans")

    if patient_data.get("stress_level", 0) >= 7:
        recommendations["lifestyle"].append("Técnicas de manejo del estrés: meditación, yoga, terapia")

    if patient_data.get("sleep_hours", 0) < 6:
        recommendations["lifestyle"].append("Mejorar higiene del sueño - objetivo: 7-9 horas por noche")

    if risks.get("diabetes_risk", 0) >= 60:
        recommendations["medical"].append("Evaluación completa de diabetes - hemoglobina A1C")
    if risks.get("cardiovascular_risk", 0) >= 60:
        recommendations["medical"].append("Evaluación cardiovascular completa - ECG, ecocardiograma")
    if risks.get("kidney_disease_risk", 0) >= 60:
        recommendations["medical"].append("Evaluación de función renal - creatinina, tasa de filtración")

    recommendations["monitoring"].append("Chequeo médico anual completo")
    recommendations["monitoring"].append("Análisis de sangre cada 6-12 meses")
    return recommendations

# ===== Helper: crear y guardar predicción =====
def _create_and_store_prediction(db: Session, user_id: str, patient_record_id: str, patient_data: dict) -> Prediction:
    """
    Llama a predict_risks(patient_data) -> obtiene dict con keys como 'diabetes_risk'...
    Crea un objeto Prediction y lo guarda en DB, retornando la instancia creada.
    Lanza excepciones hacia el caller si ocurre un error crítico.
    """
    try:
        # precondición (opcional): asegurar valores numéricos válidos
        risks = predict_risks(patient_data)  # se espera dict con percentiles 0-100
    except Exception as e:
        logger.exception("Error al predecir riesgos")
        raise

    try:
        overall_risk = calculate_overall_risk(risks)
        recommendations = generate_recommendations(patient_data, risks)

        # Crear prediction (si tu modelo SQLAlchemy tiene defaults, ajusta los campos)
        prediction = Prediction(
            id=str(uuid4()),
            user_id=user_id,
            patient_record_id=patient_record_id,
            diabetes_risk=float(risks.get("diabetes_risk", 0.0)),
            hypertension_risk=float(risks.get("hypertension_risk", 0.0)),
            cardiovascular_risk=float(risks.get("cardiovascular_risk", 0.0)),
            kidney_disease_risk=float(risks.get("kidney_disease_risk", 0.0)),
            obesity_risk=float(risks.get("obesity_risk", 0.0)),
            dyslipidemia_risk=float(risks.get("dyslipidemia_risk", 0.0)),
            metabolic_syndrome_risk=float(risks.get("metabolic_syndrome_risk", 0.0)),
            epoc_risk=float(risks.get("epoc_risk", 0.0)),
            arthritis_risk=float(risks.get("arthritis_risk", 0.0)),
            hepatopathy_risk=float(risks.get("hepatopathy_risk", 0.0)),
            parkinson_risk=float(risks.get("parkinson_risk", 0.0)),
            heart_failure_risk=float(risks.get("heart_failure_risk", 0.0)),
            alzheimer_risk=float(risks.get("alzheimer_risk", 0.0)),
            cancer_risk=float(risks.get("cancer_risk", 0.0)),
            overall_risk=overall_risk,
            recommendations=recommendations
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        logger.info(f"Prediction created: {prediction.id} for patient_record {patient_record_id}")
        return prediction
    except Exception:
        logger.exception("Error guardando la predicción en la base de datos")
        db.rollback()
        raise

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
        logger.exception("Error en registro")
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
    patient_record = PatientRecord(id=str(uuid4()), user_id=current_user.id, bmi=bmi, **record_dict)
    db.add(patient_record)
    db.commit()
    db.refresh(patient_record)
    return patient_record

@api_router.get("/patient-records", response_model=List[PatientRecordResponse])
async def get_patient_records(
    owner_roles: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role in ("admin", "medico"):
        query = db.query(PatientRecord)
        if owner_roles:
            roles = [r.strip() for r in owner_roles.split(',') if r.strip()]
            if roles:
                query = query.join(User, PatientRecord.user_id == User.id).filter(User.role.in_(roles))
        if user_id:
            query = query.filter(PatientRecord.user_id == user_id)
        records = query.order_by(PatientRecord.created_at.desc()).all()
    else:
        records = db.query(PatientRecord) \
            .filter(PatientRecord.user_id == current_user.id) \
            .order_by(PatientRecord.created_at.desc()).all()
    return records

@api_router.get("/patient-records/count")
async def get_patient_records_count(
    owner_roles: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role in ("admin", "medico"):
        q = db.query(PatientRecord)
        if owner_roles:
            roles = [r.strip() for r in owner_roles.split(',') if r.strip()]
            if roles:
                q = q.join(User, PatientRecord.user_id == User.id).filter(User.role.in_(roles))
        if user_id:
            q = q.filter(PatientRecord.user_id == user_id)
        count = q.count()
    else:
        count = db.query(PatientRecord).filter(PatientRecord.user_id == current_user.id).count()
    return {"count": count}

@api_router.get("/patient-records/{record_id}", response_model=PatientRecordResponse)
async def get_patient_record(record_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role in ("admin", "medico"):
        record = db.query(PatientRecord).filter(PatientRecord.id == record_id).first()
    else:
        record = db.query(PatientRecord).filter(PatientRecord.id == record_id, PatientRecord.user_id == current_user.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

@api_router.get("/patient-records/with-user", response_model=List[PatientRecordWithUser])
async def get_patient_records_with_user(
    owner_roles: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = db.query(PatientRecord, User).join(User, PatientRecord.user_id == User.id)

    if current_user.role in ("admin", "medico"):
        if owner_roles:
            roles = [r.strip() for r in owner_roles.split(",") if r.strip()]
            if roles:
                q = q.filter(User.role.in_(roles))
        if user_id:
            q = q.filter(PatientRecord.user_id == user_id)
    else:
        q = q.filter(PatientRecord.user_id == current_user.id)

    rows = q.order_by(desc(PatientRecord.created_at)).all()

    result: List[PatientRecordWithUser] = []
    for pr, u in rows:
        result.append(PatientRecordWithUser(
            id=pr.id,
            user_id=pr.user_id,
            age=pr.age,
            gender=pr.gender,
            bmi=pr.bmi,
            systolic_bp=pr.systolic_bp,
            diastolic_bp=pr.diastolic_bp,
            heart_rate=pr.heart_rate,
            glucose=pr.glucose,
            cholesterol_total=pr.cholesterol_total,
            cholesterol_hdl=pr.cholesterol_hdl,
            cholesterol_ldl=pr.cholesterol_ldl,
            triglycerides=pr.triglycerides,
            smoking=pr.smoking,
            alcohol_consumption=pr.alcohol_consumption,
            exercise_hours_per_week=pr.exercise_hours_per_week,
            stress_level=pr.stress_level,
            sleep_hours=pr.sleep_hours,
            family_history_diabetes=pr.family_history_diabetes,
            family_history_hypertension=pr.family_history_hypertension,
            family_history_heart_disease=pr.family_history_heart_disease,
            created_at=pr.created_at,
            user_number=getattr(u, "user_number", None),
            full_name=getattr(u, "full_name", None),
            is_active=getattr(u, "is_active", None),
            role=getattr(u, "role", None),
        ))
    return result

@api_router.post("/predictions/{record_id}", response_model=PredictionResponse)
async def create_prediction(record_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role in ("admin", "medico"):
        patient_record = db.query(PatientRecord).filter(PatientRecord.id == record_id).first()
    else:
        patient_record = db.query(PatientRecord).filter(PatientRecord.id == record_id, PatientRecord.user_id == current_user.id).first()
    if not patient_record:
        raise HTTPException(status_code=404, detail="Patient record not found")

    patient_data = {
        "age": patient_record.age,
        "gender": patient_record.gender,
        "bmi": patient_record.bmi,
        "systolic_bp": patient_record.systolic_bp,
        "diastolic_bp": patient_record.diastolic_bp,
        "heart_rate": patient_record.heart_rate,
        "glucose": patient_record.glucose,
        "cholesterol_total": patient_record.cholesterol_total,
        "cholesterol_hdl": patient_record.cholesterol_hdl,
        "cholesterol_ldl": patient_record.cholesterol_ldl,
        "triglycerides": patient_record.triglycerides,
        "smoking": patient_record.smoking,
        "alcohol_consumption": patient_record.alcohol_consumption,
        "exercise_hours_per_week": patient_record.exercise_hours_per_week,
        "stress_level": patient_record.stress_level,
        "sleep_hours": patient_record.sleep_hours,
        "family_history_diabetes": patient_record.family_history_diabetes,
        "family_history_hypertension": patient_record.family_history_hypertension,
        "family_history_heart_disease": patient_record.family_history_heart_disease,
    }

    try:
        prediction = _create_and_store_prediction(db=db, user_id=current_user.id, patient_record_id=record_id, patient_data=patient_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error creando la predicción")
        raise HTTPException(status_code=500, detail=str(e))

    return prediction

@api_router.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    user_id: Optional[str] = None,
    record_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role in ("admin", "medico"):
        q = db.query(Prediction)
        if user_id:
            q = q.filter(Prediction.user_id == user_id)
        if record_id:
            q = q.filter(Prediction.patient_record_id == record_id)
        predictions = q.order_by(Prediction.created_at.desc()).all()
    else:
        predictions = db.query(Prediction) \
            .filter(Prediction.user_id == current_user.id) \
            .order_by(Prediction.created_at.desc()).all()
    return predictions

@api_router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role in ("admin", "medico"):
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    else:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@api_router.get("/predictions/with-user", response_model=List[PredictionWithUserResponse])
async def get_predictions_with_user(
    user_id: Optional[str] = None,
    record_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = db.query(Prediction, User).join(User, Prediction.user_id == User.id)
    if current_user.role in ("admin", "medico"):
        if user_id:
            q = q.filter(Prediction.user_id == user_id)
        if record_id:
            q = q.filter(Prediction.patient_record_id == record_id)
    else:
        q = q.filter(Prediction.user_id == current_user.id)
    rows = q.order_by(desc(Prediction.created_at)).all()

    out: List[PredictionWithUserResponse] = []
    for p, u in rows:
        out.append(PredictionWithUserResponse(
            id=p.id,
            user_id=p.user_id,
            patient_record_id=p.patient_record_id,
            overall_risk=p.overall_risk,
            created_at=p.created_at,
            patient_name=getattr(u, "full_name", None),
            patient_email=getattr(u, "email", None),
            role=getattr(u, "role", None),
        ))
    return out


@api_router.post("/data/generate")
async def generate_data(num_samples: int = 10000, db: Session = Depends(get_db)):
    try:
        result = generate_synthetic_data(db, num_samples)
        return result
    except Exception as e:
        logger.exception("Error generating data")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/models/train")
async def train_models(db: Session = Depends(get_db)):
    try:
        results = train_all_models(db)
        return {"message": "All models trained successfully", "results": results}
    except Exception as e:
        logger.exception("Error training models")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/reports/pdf/{prediction_id}")
async def get_pdf_report(prediction_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role in ("admin", "medico"):
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    else:
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == current_user.id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    patient_record = db.query(PatientRecord).filter(PatientRecord.id == prediction.patient_record_id).first()
    user = current_user
    pdf_path = generate_pdf_report(user, patient_record, prediction)
    return FileResponse(path=pdf_path, filename=os.path.basename(pdf_path), media_type="application/pdf")

@api_router.get("/users")
async def list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    q = db.query(User)
    if current_user.role not in ("admin", "medico"):
        q = q.filter(User.id == current_user.id)
    # Order by existing field (user_number) desc to approximate recency
    users = q.order_by(desc(User.user_number)).all()
    out = []
    for u in users:
        out.append({
            "id": u.id,
            "email": u.email,
            "full_name": getattr(u, "full_name", None),
            "role": getattr(u, "role", None),
            "role_id": getattr(u, "role_id", None),
            "is_active": getattr(u, "is_active", None),
            "user_number": getattr(u, "user_number", None),
            "created_at": getattr(u, "created_at", None),
        })
    return out

@api_router.get("/users/count")
async def users_count(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role in ("admin", "medico"):
        count = db.query(User).count()
    else:
        count = 1
    return {"count": count}

@api_router.patch("/users/{user_id}")
async def update_user(user_id: str, payload: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.role not in ("admin", "medico"):
        raise HTTPException(status_code=403, detail="Not authorized")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Actualizar campos si se proporcionan
    if payload.full_name is not None:
        user.full_name = payload.full_name
    
    if payload.email is not None:
        # Verificar que el email no esté en uso por otro usuario
        existing_user = db.query(User).filter(User.email == payload.email, User.id != user_id).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = payload.email
    
    if payload.role_id is not None:
        # Verificar que el role_id exista
        if payload.role_id not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Invalid role_id. Must be 1 (paciente), 2 (medico), or 3 (admin)")
        user.role_id = payload.role_id
    
    if payload.is_active is not None:
        user.is_active = bool(payload.is_active)
    
    db.commit()
    db.refresh(user)
    
    return {
        "id": user.id,
        "user_number": getattr(user, "user_number", None),
        "full_name": getattr(user, "full_name", None),
        "email": getattr(user, "email", None),
        "is_active": user.is_active,
        "role": getattr(user, "role", None),
        "role_id": getattr(user, "role_id", None),
        "created_at": getattr(user, "created_at", None),
    }

# include router and shutdown
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
