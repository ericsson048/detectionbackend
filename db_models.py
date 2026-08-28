# --- ENTRAÎNEMENT ET SAUVEGARDE DU MODÈLE ---


from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, LargeBinary, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
from pydantic import BaseModel



class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

class AuthResponse(BaseModel):
    token: str
    user: UserResponse

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    # ===== RGPD / consentement =====
    consent_health_data = Column(Boolean, default=False)   # autorise le stockage des données de santé
    consent_at = Column(DateTime, nullable=True)           # date du consentement
    anonymized = Column(Boolean, default=False)            # compte anonymisé (droits RGPD)
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    reset_tokens = relationship("PasswordResetOTP", back_populates="user", cascade="all, delete-orphan")
    case_files = relationship("CaseFile", back_populates="user", cascade="all, delete-orphan")
    reminders = relationship("Reminder", back_populates="user", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    advice = Column(String)
    image = Column(LargeBinary)  # stocker l'image si besoin
    notes = Column(String, nullable=True)  # Notes de suivi patient / évolution
    # Suivi d'évolution : dossier de maladie auquel appartient cette prédiction
    case_file_id = Column(Integer, ForeignKey("case_files.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="predictions")
    case_file = relationship("CaseFile", back_populates="predictions")

class CaseFile(Base):
    """Dossier de maladie : regroupe plusieurs prédictions/photos pour suivre l'évolution."""
    __tablename__ = "case_files"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)            # titre / localisation de la lésion
    condition = Column(String, nullable=True)          # affection principale (ex. "measles")
    status = Column(String, default="stable")          # improvement / stable / worsening
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="case_files")
    predictions = relationship("Prediction", back_populates="case_file", cascade="all, delete-orphan")

class Reminder(Base):
    """Rappel personnalisé programmé pour réévaluer la peau ou prendre un soin."""
    __tablename__ = "reminders"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=True)
    remind_at = Column(DateTime, nullable=False)       # prochaine échéance
    frequency = Column(String, default="once")         # once / daily / weekly
    status = Column(String, default="active")          # active / sent / cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="reminders")

class PasswordResetOTP(Base):
    __tablename__ = "password_reset_otps"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    otp_code = Column(String(6), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="reset_tokens")
