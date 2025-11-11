# app/models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base  # <- IMPORT RELATIVO CORRECTO
import uuid
from pathlib import Path

CSV_PATH = Path(__file__).parent / "synthetic_data_full.csv"

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_number = Column(Integer, autoincrement=True, unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)

    patient_records = relationship("PatientRecord", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")


class PatientRecord(Base):
    __tablename__ = "patient_records"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)

    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    bmi = Column(Float, nullable=False)

    systolic_bp = Column(Float, nullable=False)
    diastolic_bp = Column(Float, nullable=False)
    heart_rate = Column(Float, nullable=False)

    glucose = Column(Float, nullable=False)
    cholesterol_total = Column(Float, nullable=False)
    cholesterol_hdl = Column(Float, nullable=False)
    cholesterol_ldl = Column(Float, nullable=False)
    triglycerides = Column(Float, nullable=False)

    smoking = Column(Boolean, default=False)
    alcohol_consumption = Column(String(20), default="none")
    exercise_hours_per_week = Column(Float, default=0.0)
    stress_level = Column(Integer, default=5)
    sleep_hours = Column(Float, default=7.0)

    family_history_diabetes = Column(Boolean, default=False)
    family_history_hypertension = Column(Boolean, default=False)
    family_history_heart_disease = Column(Boolean, default=False)

    current_medications = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="patient_records")
    predictions = relationship("Prediction", back_populates="patient_record", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    patient_record_id = Column(String(36), ForeignKey("patient_records.id"), nullable=False)

    diabetes_risk = Column(Float, nullable=False)
    hypertension_risk = Column(Float, nullable=False)
    cardiovascular_risk = Column(Float, nullable=False)
    kidney_disease_risk = Column(Float, nullable=False)
    obesity_risk = Column(Float, nullable=False)
    dyslipidemia_risk = Column(Float, nullable=False)
    metabolic_syndrome_risk = Column(Float, nullable=False)

    overall_risk = Column(String(20), nullable=False)
    recommendations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="predictions")
    patient_record = relationship("PatientRecord", back_populates="predictions")


class SyntheticData(Base):
    __tablename__ = "synthetic_data"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)

    bmi = Column(Float, nullable=False)

    systolic_bp = Column(Float, nullable=False)
    diastolic_bp = Column(Float, nullable=False)
    heart_rate = Column(Float, nullable=False)

    glucose = Column(Float, nullable=False)
    cholesterol_total = Column(Float, nullable=False)
    cholesterol_hdl = Column(Float, nullable=False)
    cholesterol_ldl = Column(Float, nullable=False)
    triglycerides = Column(Float, nullable=False)

    smoking = Column(Boolean, default=False)
    alcohol_consumption = Column(String(20), default="none")
    exercise_hours_per_week = Column(Float, default=0.0)
    stress_level = Column(Integer, default=5)
    sleep_hours = Column(Float, default=7.0)

    family_history_diabetes = Column(Boolean, default=False)
    family_history_hypertension = Column(Boolean, default=False)
    family_history_heart_disease = Column(Boolean, default=False)

    has_diabetes = Column(Boolean, default=False)
    has_hypertension = Column(Boolean, default=False)
    has_cardiovascular_disease = Column(Boolean, default=False)
    has_kidney_disease = Column(Boolean, default=False)
    has_obesity = Column(Boolean, default=False)
    has_dyslipidemia = Column(Boolean, default=False)
    has_metabolic_syndrome = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
