from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True

class PredictionCreate(BaseModel):
    filename: str
    prediction: str
    confidence: float
    advice: str
    image: Optional[bytes] = None

class PredictionOut(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    advice: str
    timestamp: datetime
    user_id: int

    class Config:
        from_attributes = True

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
