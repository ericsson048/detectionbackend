from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from PIL import Image
import io
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import jwt
import datetime

# ===== IMPORTS SANS PyTorch =====
import database, db_models, schemas, crud, auth
from db_models import LoginRequest, AuthResponse

# ===== CONFIGURATION =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = "dantylanez et ericsson"
ALGORITHM = "HS256"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY non configurée. Conseils IA désactivés.")

# ===== INITIALISATION =====
db_models.Base.metadata.create_all(bind=database.engine)
app = FastAPI(
    title="API Détection de maladies cutanées - Version Légère",
    description="""
    API backend pour l'application de détection de maladies cutanées.
    
    Fonctionnalités principales :
    * **Authentification** : Gestion des utilisateurs (inscription, connexion, JWT).
    * **Prédiction** : Réception des résultats de l'IA mobile et génération de conseils via Gemini.
    * **Historique** : Suivi des prédictions passées.
    
    Note : L'inférence ML principale est effectuée côté client (Flutter) avec TFLite.
    """,
    version="1.0.0"
)
get_db = database.get_db

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== FONCTIONS UTILITAIRES =====
def clean_markdown_for_mobile(text: str) -> str:
    """Convertit le markdown en texte formaté pour mobile"""
    text = re.sub(r'#{1,6}\s+(.+)', r'📌 \1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'• \1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^\s*[\*\-]\s+', '  • ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_token(user):
    """Génère un JWT pour l'utilisateur"""
    payload = {
        "userid": user.id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ===== ENDPOINTS AUTHENTIFICATION =====
@app.post("/users/", response_model=AuthResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Crée un nouvel utilisateur"""
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    user = crud.create_user(db, user)
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        },
        "token": get_token(user)
    }

@app.post("/login/")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Authentifie un utilisateur"""
    user = crud.get_user_by_username(db, data.username)
    
    if not user or not auth.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        },
        "token": get_token(user)
    }

# ===== ENDPOINT PRÉDICTION (SIMPLIFIÉ) =====
@app.post("/predict/")
async def predict(
    user_id: int = Form(...),
    prediction: str = Form(...),      # Vient de Flutter
    confidence: float = Form(...),    # Vient de Flutter
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Reçoit la prédiction calculée par Flutter.
    Génère des conseils avec Gemini (optionnel).
    Sauvegarde dans la base de données.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ===== GÉNÉRATION DE CONSEILS AVEC GEMINI =====
        final_advice = ""
        
        if GEMINI_API_KEY:
            try:
                gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                prompt = f"""
                En tant qu'assistant de santé virtuel, analyse l'image de peau ci-jointe.
                Le diagnostic suggère qu'il pourrait s'agir de : '{prediction}'.

                Fournis des conseils clairs et structurés en français :
                1. **Description brève** : Décris brièvement en une phrase ce que '{prediction}' implique.
                2. **Recommandations** : Donne 2 ou 3 conseils pratiques (hygiène, gestes à éviter, etc.).
                3. **Avertissement** : Termine TOUJOURS en rappelant que tu n'es pas un médecin et qu'il est 
                   impératif de consulter un professionnel de santé pour un diagnostic confirmé.
                
                Adopte un ton rassurant mais professionnel. Ne pose pas de question en retour.
                """

                response = gemini_model.generate_content([prompt, image])
                final_advice = clean_markdown_for_mobile(response.text)
                
            except Exception as gemini_error:
                print(f"❌ Erreur Gemini : {gemini_error}")
                final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
        else:
            final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
        
        # ===== SAUVEGARDE DANS LA BASE DE DONNÉES =====
        prediction_data = schemas.PredictionCreate(
            filename=file.filename,
            prediction=prediction,
            confidence=confidence,
            advice=final_advice,
            image=contents
        )
        crud.create_prediction(db, prediction_data, user_id)

        return JSONResponse({
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "advice": final_advice
        })
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ===== ENDPOINTS HISTORIQUE =====
@app.get("/history/{user_id}", response_model=List[schemas.PredictionOut])
def history(user_id: int, db: Session = Depends(get_db)):
    """Récupère l'historique des prédictions d'un utilisateur"""
    return crud.get_predictions(db, user_id)

@app.get("/predictions/unsynced/{user_id}", response_model=List[schemas.PredictionOut])
def unsynced_predictions(user_id: int, db: Session = Depends(get_db)):
    """Récupère les prédictions non synchronisées"""
    return db.query(db_models.Prediction).filter(
        db_models.Prediction.user_id == user_id
    ).all()

# ===== ENDPOINTS UTILISATEURS =====
@app.get("/users/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son ID"""
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/username/{username}", response_model=schemas.UserOut)
def get_user_by_username(username: str, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son nom d'utilisateur"""
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/", response_model=List[schemas.UserOut])
def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Récupère tous les utilisateurs (avec pagination)"""
    return crud.get_users(db, skip=skip, limit=limit)

# ===== HEALTH CHECK =====
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "API is running - Version légère (sans PyTorch)",
        "ml_inference": "Client-side (Flutter)"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "memory_usage": "< 200MB"}


# ===== KEEP-ALIVE POUR RENDER =====
import threading
import time
import requests

def keep_alive():
    """Ping l'API toutes les 2 minutes pour éviter la mise en veille (Render free)"""
    while True:
        try:
            url = "https://detectionbackend-hln7.onrender.com"
            if url:
                print(f"🔄 Keep-alive ping vers {url}")
                requests.get(url + "/health", timeout=2)
        except Exception as e:
            print(f"⚠️ Keep-alive error: {e}")
        
        time.sleep(120)  # 2 minutes

# Lancer le keep-alive dans un thread séparé
threading.Thread(target=keep_alive, daemon=True).start()


# ===== DÉMARRAGE =====
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)  # Pas de --reload en production !