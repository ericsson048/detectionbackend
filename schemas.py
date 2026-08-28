from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    consent_health_data: Optional[bool] = False

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    consent_health_data: Optional[bool] = False
    anonymized: Optional[bool] = False

    class Config:
        from_attributes = True

class PredictionCreate(BaseModel):
    filename: str
    prediction: str
    confidence: float
    advice: str
    image: Optional[bytes] = None
    case_file_id: Optional[int] = None

class PredictionOut(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    advice: str
    notes: Optional[str] = None
    case_file_id: Optional[int] = None
    timestamp: datetime
    user_id: int

    class Config:
        from_attributes = True

class PredictionNotesUpdate(BaseModel):
    notes: str

# ===== Suivi d'évolution (dossier de maladie) =====
class CaseFileCreate(BaseModel):
    title: str
    condition: Optional[str] = None
    status: Optional[str] = "stable"

class CaseFileUpdate(BaseModel):
    title: Optional[str] = None
    condition: Optional[str] = None
    status: Optional[str] = None

class CaseFileOut(BaseModel):
    id: int
    user_id: int
    title: str
    condition: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    predictions: List[PredictionOut] = []

    class Config:
        from_attributes = True

# ===== Questionnaire anamnestique =====
class AnamnesisRequest(BaseModel):
    user_id: int
    prediction: str
    confidence: float
    questions: dict   # { "itchy": bool, "pain": bool, "duration_days": int, "history": str, ... }

class AnamnesisResponse(BaseModel):
    refined_advice: str

# ===== Qualité d'image =====
class ImageQualityIssueSchema(BaseModel):
    code: str
    title: str
    description: str
    suggestion: str

class ImageQualityResponse(BaseModel):
    usable: bool
    average_brightness: float
    sharpness_variance: float
    issues: List[ImageQualityIssueSchema]

# ===== PDF côté serveur =====
class PdfRequest(BaseModel):
    user_id: int
    case_file_id: Optional[int] = None   # si fourni : rapport du dossier complet
    prediction_id: Optional[int] = None  # si fourni : rapport d'une seule prédiction
    send_email_to: Optional[EmailStr] = None

class PdfResponse(BaseModel):
    pdf_base64: Optional[str] = None
    emailed_to: Optional[str] = None

# ===== Rappels personnalisés =====
class ReminderCreate(BaseModel):
    user_id: int
    title: str
    message: Optional[str] = None
    remind_at: datetime
    frequency: Optional[str] = "once"

class ReminderUpdate(BaseModel):
    title: Optional[str] = None
    message: Optional[str] = None
    remind_at: Optional[datetime] = None
    frequency: Optional[str] = None
    status: Optional[str] = None

class ReminderOut(BaseModel):
    id: int
    user_id: int
    title: str
    message: Optional[str] = None
    remind_at: datetime
    frequency: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

# ===== RGPD =====
class ConsentUpdate(BaseModel):
    consent_health_data: bool

class ExportOut(BaseModel):
    user_id: int
    username: str
    email: str
    consent_health_data: bool
    anonymized: bool
    predictions_count: int
    case_files_count: int
    predictions: List[PredictionOut] = []
    case_files: List[CaseFileOut] = []

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp_code: str
    new_password: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str

class StatsOverview(BaseModel):
    total_users: int
    total_predictions: int
    conditions_distribution: dict
    healthy_ratio: float
    avg_confidence: float
