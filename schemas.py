from pydantic import BaseModel
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
        orm_mode = True

class PredictionCreate(BaseModel):
    filename: str
    prediction: str
    confidence: float
    advice: str
    image: Optional[bytes] = None

class PredictionOut(PredictionCreate):
    id: int
    timestamp: datetime
    user_id: int

    class Config:
        orm_mode = True
