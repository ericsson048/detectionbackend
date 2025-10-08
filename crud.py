from sqlalchemy.orm import Session
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