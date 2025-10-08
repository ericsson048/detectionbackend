from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from models import predict_image, MODEL_PATH, create_model
import os
import torch

# --- INITIALISATION FASTAPI ---
app = FastAPI(title="Mpox Detection API")

# Vérifier si le modèle local existe, sinon le créer via model.py
if not os.path.exists(MODEL_PATH):
    from models import train_and_save_model  # Si tu veux entraîner ici
    train_and_save_model()  # Fonction à créer dans model.py si nécessaire

# --- ROUTE DE PRÉDICTION ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        prediction = predict_image(image_bytes)
        return JSONResponse({"prediction": prediction})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
from typing import List

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    try:
        for file in files:
            image_bytes = await file.read()
            prediction = predict_image(image_bytes)
            results.append({"filename": file.filename, "prediction": prediction})
        return JSONResponse({"predictions": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/retrain")
def retrain_model():
    from models import train_and_save_model
    try:
        train_and_save_model()
        return {"status": "success", "message": "Le modèle a été réentraîné et sauvegardé."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/retrain")
def retrain_model():
    from models import train_and_save_model
    try:
        train_and_save_model()
        return {"status": "success", "message": "Le modèle a été réentraîné et sauvegardé."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
# @app.get("/model-info")
# def model_info():
#     return {
#         "model_name": "ViT Base Patch16 224",
#         "device": str(device),
#         "num_classes": len(classes)
#     }


# --- ROUTE HEALTH CHECK ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- LANCEMENT Uvicorn ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
