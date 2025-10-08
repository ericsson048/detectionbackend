# --- ENTRAÎNEMENT ET SAUVEGARDE DU MODÈLE ---


from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    advice = Column(String)
    image = Column(LargeBinary)  # stocker l'image si besoin
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="predictions")
