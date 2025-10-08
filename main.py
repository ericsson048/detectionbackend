from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models as torch_models, transforms  # Renommé pour éviter la confusion

import database, db_models, schemas, crud, auth  # Retiré 'models' et ajouté 'db_models'

from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
from fastapi.middleware.cors import CORSMiddleware





# ------------------- Initialisation -------------------
db_models.Base.metadata.create_all(bind=database.engine)  # Utilise db_models au lieu de models
app = FastAPI(title="API Détection de maladies cutanées")
get_db = database.get_db

# ------------------- Modèle -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch_models.resnet18(weights=None)  # Utilise torch_models au lieu de models
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("best_mpox_model.pth", map_location=device))
model = model.to(device)
model.eval()

classes = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

advice_dict = {
    "Healthy": "Continuez à maintenir une bonne hygiène de la peau.",
    "Monkeypox": "Isolez-vous et consultez un médecin rapidement.",
    "Measles": "Consultez un médecin et surveillez la fièvre.",
    "Chickenpox": "Isolez-vous et surveillez l'évolution des éruptions cutanées.",
    "Cowpox": "Consultez un médecin et évitez le contact avec les animaux.",
    "HFMD": "Lavez-vous les mains fréquemment et surveillez les symptômes."
}


### AJOUT GEMINI : Configuration ###
# Charger les variables d'environnement à partir du fichier .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vérifier si la clé API est bien chargée et configurer l'API
if not GEMINI_API_KEY:
    print("AVERTISSEMENT: La clé GEMINI_API_KEY n'est pas trouvée. Les conseils de l'IA seront désactivés.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
# ------------------------------------

def clean_markdown_for_mobile(text: str) -> str:
    """Convertit le markdown en texte formaté pour React Native"""
    # Remplacer les titres avec des emojis
    text = re.sub(r'#{1,6}\s+(.+)', r'📌 \1', text)
    
    # Remplacer le gras **texte** 
    text = re.sub(r'\*\*(.+?)\*\*', r'• \1', text)
    
    # Remplacer l'italique
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    # Nettoyer les listes à puces
    text = re.sub(r'^\s*[\*\-]\s+', '  • ', text, flags=re.MULTILINE)
    
    # Nettoyer les lignes vides multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Users -------------------
@app.post("/users/", response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return crud.create_user(db, user)

# ------------------- Login simple -------------------
@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, username)
    if not user or not auth.verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user.id, "username": user.username}

# ------------------- Prédiction -------------------
@app.post("/predict/")
async def predict(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted_class = classes[torch.argmax(probs)]
            confidence = float(probs.max())
            
        ### AJOUT GEMINI : Génération de conseil dynamique ###
        final_advice = ""
        # On tente d'appeler Gemini seulement si la clé API a été chargée
        if GEMINI_API_KEY:
            try:
                gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                # Le prompt guide l'IA pour qu'elle donne une réponse utile
                prompt = f"""
                En tant qu'assistant de santé virtuel, analyse l'image de peau ci-jointe.
                Mon premier diagnostic suggère qu'il pourrait s'agir de : '{predicted_class}'.

                Fournis des conseils clairs et structurés en français.
                1.  **Description brève** : Décris brièvement en une phrase ce que '{predicted_class}' implique.
                2.  **Recommandations** : Donne 2 ou 3 conseils pratiques (hygiène, gestes à éviter, etc.).
                3.  **Avertissement** : Termine TOUJOURS en rappelant que tu n'es pas un médecin et qu'il est impératif de consulter un professionnel de santé pour un diagnostic confirmé.
                
                Adopte un ton rassurant mais professionnel. Ne pose pas de question en retour.
                """

                # On envoie l'image et le prompt à Gemini
                # Il faut passer une image PIL directement à la librairie
                response = gemini_model.generate_content([prompt, image])
                final_advice = clean_markdown_for_mobile(response.text)

            except Exception as gemini_error:
                print(f"Erreur lors de l'appel à l'API Gemini : {gemini_error}")
                # En cas d'erreur avec Gemini, on utilise le dictionnaire de secours
                final_advice = advice_dict.get(predicted_class, "Consultez un professionnel de santé pour un avis personnalisé.")
        else:
            # Si la clé API n'a jamais été configurée, on utilise directement le dictionnaire
            final_advice = advice_dict.get(predicted_class, "Consultez un professionnel de santé pour un avis personnalisé.")
        # ----------------------------------------------------
        
         # Sauvegarde dans la DB
        prediction_data = schemas.PredictionCreate(
            filename=file.filename,
            prediction=predicted_class,
            confidence=confidence,
            advice=final_advice,
            image=contents
        )
        crud.create_prediction(db, prediction_data, user_id)

        return JSONResponse({
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": confidence,
            "advice": final_advice
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ------------------- Historique -------------------
@app.get("/history/{user_id}", response_model=List[schemas.PredictionOut])
def history(user_id: int, db: Session = Depends(get_db)):
    return crud.get_predictions(db, user_id)

# ------------------- Prédictions non synchronisées -------------------
@app.get("/predictions/unsynced/{user_id}", response_model=List[schemas.PredictionOut])
def unsynced_predictions(user_id: int, db: Session = Depends(get_db)):
    # Filtre pour synced=False si vous ajoutez ce champ pour offline
    return db.query(db_models.Prediction).filter(db_models.Prediction.user_id==user_id).all()  # Utilise db_models

# # ------------------- Users -------------------
# @app.post("/users/", response_model=schemas.UserOut)
# def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     db_user = crud.get_user_by_username(db, user.username)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Username already registered")
#     return crud.create_user(db, user)

# ------------------- Get User by ID -------------------
@app.get("/users/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son ID"""
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ------------------- Get User by Username -------------------
@app.get("/users/username/{username}", response_model=schemas.UserOut)
def get_user_by_username(username: str, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son nom d'utilisateur"""
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ------------------- Get All Users (optionnel - pour admin) -------------------
@app.get("/users/", response_model=List[schemas.UserOut])
def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Récupère tous les utilisateurs (avec pagination)"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)