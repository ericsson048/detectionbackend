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
import time
from functools import wraps

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


def rate_limit_retry(max_retries=3, initial_delay=2):
    """Décorateur pour gérer les rate limits avec retry exponentiel"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "RATE_LIMIT_EXCEEDED" in str(e):
                        if attempt < max_retries - 1:
                            print(f"⏳ Rate limit atteint, attente {delay}s (tentative {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            delay *= 2  # Backoff exponentiel
                        else:
                            raise
                    else:
                        raise
            return None
        return wrapper
    return decorator

# ===== ENDPOINT PRÉDICTION (SIMPLIFIÉ) =====
# @app.post("/predict/")
# async def predict(
#     user_id: int = Form(...),
#     prediction: str = Form(...),
#     confidence: float = Form(...),
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         final_advice = ""

#         # logger.info("Prediction request received",image)
        
#         if GEMINI_API_KEY:
#             try:
#                 # CHANGEZ LE MODÈLE : utilisez gemini-1.5-flash (plus de quota)
#                 gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
#                 # Configuration de sécurité pour images médicales
#                 safety_settings = [
#                     {
#                         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                         "threshold": "BLOCK_ONLY_HIGH"
#                     },
#                     {
#                         "category": "HARM_CATEGORY_HARASSMENT",
#                         "threshold": "BLOCK_ONLY_HIGH"
#                     },
#                     {
#                         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                         "threshold": "BLOCK_ONLY_HIGH"  # Important pour images de peau
#                     },
#                 ]
                
#                 prompt = f"""
#                 En tant qu'assistant de santé virtuel, analyse l'image de peau ci-jointe.
#                 Le diagnostic suggère qu'il pourrait s'agir de : '{prediction}'.

#                 Fournis des conseils clairs et structurés en français :
#                 1. **Description brève** : Décris brièvement en une phrase ce que '{prediction}' implique.
#                 2. **Recommandations** : Donne 2 ou 3 conseils pratiques (hygiène, gestes à éviter, etc.).
#                 3. **Avertissement** : Termine TOUJOURS en rappelant que tu n'es pas un médecin et qu'il est 
#                    impératif de consulter un professionnel de santé pour un diagnostic confirmé.
                
#                 Adopte un ton rassurant mais professionnel. Ne pose pas de question en retour.
#                 """

#                 # Ajout d'un délai entre requêtes (simple throttling)
#                 time.sleep(1)  # Attendre 1 seconde avant chaque requête
                
#                 response = gemini_model.generate_content(
#                     prompt,
#                     safety_settings=safety_settings
#                 )
#                 final_advice = clean_markdown_for_mobile(response.text)
                
#             except Exception as gemini_error:
#                 error_msg = str(gemini_error)
#                 print(f"❌ Erreur Gemini : {error_msg}")
                
#                 # Messages d'erreur plus spécifiques
#                 if "429" in error_msg or "RATE_LIMIT" in error_msg:
#                     final_advice = "⏳ Service temporairement surchargé. Réessayez dans quelques instants."
#                 elif "SAFETY" in error_msg:
#                     final_advice = "Image non analysable. Consultez un professionnel de santé."
#                 else:
#                     final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
#         else:
#             final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
        
#         # Sauvegarde dans la base (même si Gemini échoue)
#         prediction_data = schemas.PredictionCreate(
#             filename=file.filename,
#             prediction=prediction,
#             confidence=confidence,
#             advice=final_advice,
#             image=contents
#         )
#         crud.create_prediction(db, prediction_data, user_id)

#         return JSONResponse({
#             "filename": file.filename,
#             "prediction": prediction,
#             "confidence": confidence,
#             "advice": final_advice
#         })
        
#     except Exception as e:
#         print(f"❌ Erreur : {e}")
#         return JSONResponse({"error": str(e)}, status_code=500)

# @app.post("/predict/")
# async def predict(
#     user_id: int = Form(...),
#     prediction: str = Form(...),
#     confidence: float = Form(...),
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         final_advice = ""
        
#         if GEMINI_API_KEY:
#             try:
#                 # Modèle gemini-1.5-flash : bon compromis quota/performance/multimodalité
#                 gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
#                 # Configuration de sécurité pour images médicales
#                 safety_settings = [
#                     {
#                         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                         "threshold": "BLOCK_ONLY_HIGH"
#                     },
#                     {
#                         "category": "HARM_CATEGORY_HARASSMENT",
#                         "threshold": "BLOCK_ONLY_HIGH"
#                     },
#                     {
#                         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                         "threshold": "BLOCK_ONLY_HIGH"  # Important pour images de peau
#                     },
#                 ]
                
#                 prompt = f"""
#                 En tant qu'assistant de santé virtuel, analyse l'image de peau ci-jointe.
#                 Le diagnostic suggère qu'il pourrait s'agir de : '{prediction}'.

#                 Fournis des conseils clairs et structurés en français :
#                 1. **Description brève** : Décris brièvement en une phrase ce que '{prediction}' implique.
#                 2. **Recommandations** : Donne 2 ou 3 conseils pratiques (hygiène, gestes à éviter, etc.).
#                 3. **Avertissement** : Termine TOUJOURS en rappelant que tu n'es pas un médecin et qu'il est 
#                    impératif de consulter un professionnel de santé pour un diagnostic confirmé.
                
#                 Adopte un ton rassurant mais professionnel. Ne pose pas de question en retour.
#                 """

#                 # Ajout d'un délai entre requêtes (simple throttling)
#                 time.sleep(1) 
                
#                 # --- MODIFICATION CRUCIALE : Passage de l'image (objet PIL) au modèle ---
#                 response = gemini_model.generate_content(
#                     [prompt, image], # <-- L'image et le prompt sont passés ensemble
#                     safety_settings=safety_settings
#                 )
#                 final_advice = clean_markdown_for_mobile(response.text)
                
#             except Exception as gemini_error:
#                 error_msg = str(gemini_error)
#                 print(f"❌ Erreur Gemini : {error_msg}")
                
#                 # Messages d'erreur plus spécifiques
#                 if "429" in error_msg or "RATE_LIMIT" in error_msg:
#                     final_advice = "⏳ Service temporairement surchargé. Réessayez dans quelques instants."
#                 elif "SAFETY" in error_msg:
#                     # En cas de blocage de sécurité, on utilise un message d'avertissement
#                     final_advice = "Image non analysable. Consultez un professionnel de santé."
#                 else:
#                     final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
#         else:
#             final_advice = "Consultez un professionnel de santé pour un avis personnalisé."
        
#         # Sauvegarde dans la base (même si Gemini échoue)
#         prediction_data = schemas.PredictionCreate(
#             filename=file.filename,
#             prediction=prediction,
#             confidence=confidence,
#             advice=final_advice,
#             image=contents
#         )
#         crud.create_prediction(db, prediction_data, user_id)

#         return JSONResponse({
#             "filename": file.filename,
#             "prediction": prediction,
#             "confidence": confidence,
#             "advice": final_advice
#         })
        
#     except Exception as e:
#         print(f"❌ Erreur générale : {e}")
#         return JSONResponse({"error": str(e)}, status_code=500)


# ===== DICTIONNAIRE DE CONSEILS STATIQUES =====
STATIC_ADVICE = {
    "chickenpox": {
        "description": "La varicelle est une infection virale très contagieuse causant des éruptions de vésicules.",
        "recommendations": [
            "Isolez-vous pour éviter la contagion (surtout femmes enceintes et bébés)",
            "Ne grattez pas les boutons pour éviter les cicatrices et infections",
            "Appliquez des compresses fraîches et des crèmes apaisantes",
            "Coupez vos ongles courts et portez des gants la nuit si nécessaire"
        ],
        "warning": "⚠️ Consultez un médecin immédiatement si : fièvre élevée, difficultés respiratoires, ou si vous êtes immunodéprimé."
    },
    "cowpox": {
        "description": "La vaccine (cowpox) est une infection virale rare transmise par contact avec des animaux infectés.",
        "recommendations": [
            "Évitez de toucher la lésion et lavez-vous soigneusement les mains",
            "Couvrez la zone infectée avec un pansement propre",
            "Ne partagez pas vos objets personnels (serviettes, vêtements)",
            "Surveillez les signes d'infection secondaire (pus, rougeur croissante)"
        ],
        "warning": "Consultez un médecin pour confirmer le diagnostic et écarter d'autres infections virales plus graves."
    },
    "hfmd": {
        "description": "Le syndrome pieds-mains-bouche (HFMD) est une infection virale courante chez les enfants.",
        "recommendations": [
            "Hydratez-vous bien et mangez des aliments mous si la bouche est douloureuse",
            "Lavez-vous fréquemment les mains, surtout après contact avec les lésions",
            "Évitez le contact rapproché avec d'autres personnes pendant 7-10 jours",
            "Désinfectez les surfaces et jouets régulièrement"
        ],
        "warning": "Consultez un médecin si : forte fièvre, déshydratation, maux de tête sévères ou raideur de la nuque."
    },
    "healthy": {
        "description": "Aucune anomalie cutanée détectée. Votre peau semble en bonne santé !",
        "recommendations": [
            "Continuez à protéger votre peau du soleil (SPF 30+ minimum)",
            "Maintenez une bonne hydratation quotidienne",
            "Adoptez une alimentation équilibrée riche en vitamines",
            "Surveillez tout changement inhabituel (grains de beauté, taches)"
        ],
        "warning": "Consultez un dermatologue une fois par an pour un examen préventif, surtout si antécédents familiaux."
    },
    "measles": {
        "description": "La rougeole est une infection virale très contagieuse et potentiellement grave.",
        "recommendations": [
            "⚠️ ISOLATION STRICTE : Restez chez vous pendant au moins 4 jours après l'éruption",
            "Reposez-vous dans une pièce sombre (sensibilité à la lumière)",
            "Hydratez-vous abondamment et prenez du paracétamol pour la fièvre",
            "Évitez tout contact avec personnes non vaccinées, femmes enceintes et bébés"
        ],
        "warning": "⚠️ URGENCE MÉDICALE : Consultez immédiatement si complications (difficultés respiratoires, convulsions, confusion). La rougeole peut être mortelle."
    },
    "monkeypox": {
        "description": "La variole du singe (Mpox) est une infection virale avec éruption cutanée caractéristique.",
        "recommendations": [
            "⚠️ ISOLEMENT REQUIS : Évitez tout contact physique jusqu'à guérison complète",
            "Couvrez les lésions avec des pansements et changez-les régulièrement",
            "Désinfectez toutes les surfaces touchées et lavez le linge à haute température",
            "Ne partagez aucun objet personnel (literie, vêtements, ustensiles)"
        ],
        "warning": "⚠️ DÉCLARATION OBLIGATOIRE : Contactez immédiatement votre médecin ou les autorités sanitaires. Cette maladie nécessite un suivi médical strict."
    },
    "default": {
        "description": "Affection cutanée détectée nécessitant une évaluation professionnelle.",
        "recommendations": [
            "Maintenez une bonne hygiène de la peau",
            "Évitez de gratter ou irriter la zone affectée",
            "Documentez l'évolution avec des photos datées",
            "Notez tout symptôme associé (fièvre, douleur, démangeaisons)"
        ],
        "warning": "Seul un professionnel de santé peut établir un diagnostic précis. Consultez rapidement."
    }
}

def get_static_advice(prediction: str) -> str:
    """
    Génère des conseils statiques formatés basés sur la prédiction.
    Recherche par correspondance partielle (insensible à la casse).
    """
    prediction_lower = prediction.lower()
    
    # Recherche de correspondance
    advice_data = None
    for key in STATIC_ADVICE.keys():
        if key in prediction_lower or prediction_lower in key:
            advice_data = STATIC_ADVICE[key]
            break
    
    # Si aucune correspondance, utiliser les conseils par défaut
    if not advice_data:
        advice_data = STATIC_ADVICE["default"]
    
    # Formatage pour mobile
    formatted_advice = f"📌 {advice_data['description']}\n\n"
    formatted_advice += "💡 Recommandations :\n"
    for i, rec in enumerate(advice_data['recommendations'], 1):
        formatted_advice += f"  • {rec}\n"
    formatted_advice += f"\n⚠️ {advice_data['warning']}"
    
    return formatted_advice


# ===== ENDPOINT PRÉDICTION AVEC CONSEILS STATIQUES =====
@app.post("/predict/")
async def predict(
    user_id: int = Form(...),
    prediction: str = Form(...),
    confidence: float = Form(...),
    file: UploadFile = File(...),
    use_ai_enhancement: bool = Form(False),  # Paramètre optionnel pour activer Gemini
    db: Session = Depends(get_db)
):
    """
    Endpoint de prédiction avec conseils statiques fiables.
    
    Paramètres:
    - user_id: ID de l'utilisateur
    - prediction: Nom de la maladie détectée
    - confidence: Niveau de confiance (0-1)
    - file: Image analysée
    - use_ai_enhancement: Active l'enrichissement par IA (optionnel, défaut: False)
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 1. TOUJOURS générer des conseils statiques (fiables et instantanés)
        final_advice = get_static_advice(prediction)
        
        # 2. Enrichissement optionnel par Gemini (si activé ET clé API disponible)
        if use_ai_enhancement and GEMINI_API_KEY:
            try:
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                ]
                
                prompt = f"""
                Enrichis ces conseils de base pour '{prediction}' avec 1-2 informations complémentaires utiles :
                
                {final_advice}
                
                Ajoute SEULEMENT :
                - Un conseil pratique supplémentaire spécifique à cette condition
                - OU une précision sur quand consulter en urgence
                
                Reste concis (max 3 phrases). Ton rassurant et professionnel.
                """

                time.sleep(1)  # Rate limiting
                
                response = gemini_model.generate_content(
                    prompt,
                    safety_settings=safety_settings
                )
                
                # Ajouter l'enrichissement IA après les conseils statiques
                ai_enhancement = clean_markdown_for_mobile(response.text)
                final_advice += f"\n\n🤖 Complément IA :\n{ai_enhancement}"
                
            except Exception as gemini_error:
                error_msg = str(gemini_error)
                print(f"⚠️ Enrichissement IA échoué : {error_msg}")
                # On garde les conseils statiques sans échouer
                if "429" in error_msg or "RATE_LIMIT" in error_msg:
                    final_advice += "\n\n(Enrichissement IA temporairement indisponible)"
        
        # 3. Sauvegarde dans la base
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
            "advice": final_advice,
            "advice_source": "static+ai" if use_ai_enhancement and GEMINI_API_KEY else "static"
        })
        
    except Exception as e:
        print(f"❌ Erreur générale : {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ===== ENDPOINT POUR LISTER LES MALADIES SUPPORTÉES =====
@app.get("/supported-conditions/")
def get_supported_conditions():
    """Retourne la liste des conditions avec conseils statiques disponibles"""
    conditions = []
    for key, value in STATIC_ADVICE.items():
        if key != "default":
            conditions.append({
                "name": key,
                "description": value["description"]
            })
    return {"conditions": conditions, "total": len(conditions)}
    

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