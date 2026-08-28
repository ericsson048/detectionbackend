from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
import random
import db_models, schemas, auth  

# Users
def create_user(db: Session, user: schemas.UserCreate):
    hashed_pw = auth.hash_password(user.password)
    db_user = db_models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_pw,
        consent_health_data=user.consent_health_data,
        consent_at=datetime.utcnow() if user.consent_health_data else None,
        anonymized=False,
    )
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

def get_prediction_by_id(db: Session, prediction_id: int):
    return db.query(db_models.Prediction).filter(db_models.Prediction.id == prediction_id).first()

def update_prediction_notes(db: Session, prediction_id: int, notes: str):
    pred = get_prediction_by_id(db, prediction_id)
    if not pred:
        return None
    pred.notes = notes
    db.commit()
    db.refresh(pred)
    return pred

def get_user_by_email(db: Session, email: str):
    """Récupère un utilisateur par son email"""
    return db.query(db_models.User).filter(db_models.User.email == email).first()

# ==== Suivi d'évolution : dossiers de maladie ====
def create_case_file(db: Session, user_id: int, data: schemas.CaseFileCreate):
    cf = db_models.CaseFile(
        user_id=user_id,
        title=data.title,
        condition=data.condition,
        status=data.status or "stable",
    )
    db.add(cf)
    db.commit()
    db.refresh(cf)
    return cf

def get_case_files(db: Session, user_id: int):
    return db.query(db_models.CaseFile).filter(
        db_models.CaseFile.user_id == user_id
    ).order_by(db_models.CaseFile.updated_at.desc()).all()

def get_case_file(db: Session, case_file_id: int):
    return db.query(db_models.CaseFile).filter(db_models.CaseFile.id == case_file_id).first()

def update_case_file(db: Session, case_file_id: int, data: schemas.CaseFileUpdate):
    cf = get_case_file(db, case_file_id)
    if not cf:
        return None
    if data.title is not None:
        cf.title = data.title
    if data.condition is not None:
        cf.condition = data.condition
    if data.status is not None:
        cf.status = data.status
    db.commit()
    db.refresh(cf)
    return cf

def delete_case_file(db: Session, case_file_id: int):
    cf = get_case_file(db, case_file_id)
    if not cf:
        return False
    db.delete(cf)
    db.commit()
    return True

# ==== Rappels ====
def create_reminder(db: Session, data: schemas.ReminderCreate):
    r = db_models.Reminder(
        user_id=data.user_id,
        title=data.title,
        message=data.message,
        remind_at=data.remind_at,
        frequency=data.frequency or "once",
        status="active",
    )
    db.add(r)
    db.commit()
    db.refresh(r)
    return r

def get_reminders(db: Session, user_id: int):
    return db.query(db_models.Reminder).filter(
        db_models.Reminder.user_id == user_id
    ).order_by(db_models.Reminder.remind_at.asc()).all()

def get_reminder(db: Session, reminder_id: int):
    return db.query(db_models.Reminder).filter(db_models.Reminder.id == reminder_id).first()

def update_reminder(db: Session, reminder_id: int, data: schemas.ReminderUpdate):
    r = get_reminder(db, reminder_id)
    if not r:
        return None
    if data.title is not None:
        r.title = data.title
    if data.message is not None:
        r.message = data.message
    if data.remind_at is not None:
        r.remind_at = data.remind_at
    if data.frequency is not None:
        r.frequency = data.frequency
    if data.status is not None:
        r.status = data.status
    db.commit()
    db.refresh(r)
    return r

def delete_reminder(db: Session, reminder_id: int):
    r = get_reminder(db, reminder_id)
    if not r:
        return False
    db.delete(r)
    db.commit()
    return True

def get_due_reminders(db: Session, now: datetime):
    """Retourne les rappels actifs dont l'échéance est passée."""
    return db.query(db_models.Reminder).filter(
        db_models.Reminder.status == "active",
        db_models.Reminder.remind_at <= now
    ).all()

# ==== RGPD ====
def update_consent(db: Session, user_id: int, consent: bool):
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    user.consent_health_data = consent
    user.consent_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user

def anonymize_user(db: Session, user_id: int):
    """Anonymise un utilisateur : remplace données identifiantes mais garde le compte."""
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    suffix = f"anonyme_{user.id}"
    user.username = f"utilisateur_{suffix}"
    user.email = f"{suffix}@anonymise.skindetect"
    user.anonymized = True
    user.consent_health_data = False
    db.commit()
    db.refresh(user)
    return user

def delete_user(db: Session, user_id: int):
    """Supprime définitivement le compte et toutes ses données (droit à l'oubli)."""
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)  # cascade supprime prédictions, dossiers, rappels, OTPs
    db.commit()
    return True

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