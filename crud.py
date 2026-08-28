from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
import random
import db_models, schemas, auth  

# Users
def create_user(db: Session, user: schemas.UserCreate):
    hashed_pw = auth.hash_password(user.password)
    db_user = db_models.User(username=user.username, email=user.email, hashed_password=hashed_pw)  # Utilise db_models
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_username(db: Session, username: str):
    return db.query(db_models.User).filter(db_models.User.username == username).first()  # Utilise db_models

def get_user_by_id(db: Session, user_id: int):
    """Récupère un utilisateur par son ID"""
    return db.query(db_models.User).filter(db_models.User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    """Récupère tous les utilisateurs avec pagination"""
    return db.query(db_models.User).offset(skip).limit(limit).all()

# Predictions
def create_prediction(db: Session, prediction: schemas.PredictionCreate, user_id: int):
    db_pred = db_models.Prediction(**prediction.dict(), user_id=user_id)  # Utilise db_models
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_predictions(db: Session, user_id: int):
    return db.query(db_models.Prediction).filter(db_models.Prediction.user_id == user_id).order_by(db_models.Prediction.timestamp.desc()).all()  # Utilise db_models

def get_user_by_email(db: Session, email: str):
    """Récupère un utilisateur par son email"""
    return db.query(db_models.User).filter(db_models.User.email == email).first()

# Reset Password OTP
def create_reset_otp(db: Session, user_id: int) -> str:
    """Crée et retourne un OTP pour la réinitialisation de mot de passe"""
    otp_code = str(random.randint(100000, 999999))
    expires_at = datetime.utcnow() + timedelta(minutes=15)
    otp = db_models.PasswordResetOTP(
        user_id=user_id,
        otp_code=otp_code,
        expires_at=expires_at,
        used=False
    )
    db.add(otp)
    db.commit()
    db.refresh(otp)
    return otp_code

def get_valid_otp(db: Session, user_id: int, otp_code: str):
    """Retourne un OTP valide, non utilisé et non expiré"""
    return db.query(db_models.PasswordResetOTP).filter(
        db_models.PasswordResetOTP.user_id == user_id,
        db_models.PasswordResetOTP.otp_code == otp_code,
        db_models.PasswordResetOTP.used == False,
        db_models.PasswordResetOTP.expires_at > datetime.utcnow()
    ).first()

def invalidate_otp(db: Session, otp):
    """Marque un OTP comme utilisé"""
    otp.used = True
    db.commit()

def reset_user_password(db: Session, user_id: int, new_password: str):
    """Met à jour le mot de passe d'un utilisateur"""
    user = db.query(db_models.User).filter(db_models.User.id == user_id).first()
    if user:
        user.hashed_password = auth.hash_password(new_password)
        db.commit()
        return True
    return False

# Statistiques
def get_global_stats(db: Session) -> dict:
    """Aggrège les statistiques globales de prédiction"""
    total_users = db.query(db_models.User).count()
    total_predictions = db.query(db_models.Prediction).count()

    if total_predictions == 0:
        return {
            "total_users": total_users,
            "total_predictions": 0,
            "conditions_distribution": {},
            "healthy_ratio": 0.0,
            "avg_confidence": 0.0
        }

    rows = db.query(
        db_models.Prediction.prediction,
        func.count(db_models.Prediction.id)
    ).group_by(db_models.Prediction.prediction).all()

    conditions_distribution = {pred: count for pred, count in rows}

    healthy_count = sum(
        count for pred, count in rows
        if "healthy" in pred.lower()
    )

    avg_confidence = db.query(func.avg(db_models.Prediction.confidence)).scalar() or 0.0

    return {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "conditions_distribution": conditions_distribution,
        "healthy_ratio": round(healthy_count / total_predictions, 4),
        "avg_confidence": round(avg_confidence, 4)
    }