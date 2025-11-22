import os
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "ericssonish/detectionbackend-mpox" # Changez 'ericsson048' par votre nom d'utilisateur HF si différent
MODEL_PATH = "best_mpox_model.pth"
MODEL_FILE = "models.py"

def upload_to_hf():
    print("Authentification...")
    # L'utilisateur sera invité à entrer son token si pas déjà connecté
    # ou on peut utiliser login() de huggingface_hub
    from huggingface_hub import login
    # login() # Décommentez pour une authentification interactive ou utilisez un token
    print("Assurez-vous d'être authentifié via 'huggingface-cli login' ou en passant un token.")

    api = HfApi()
    
    print(f"Création du repo {REPO_ID}...")
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Erreur lors de la création du repo (il existe peut-être déjà): {e}")

    print("Génération du README.md (Model Card)...")
    readme_content = """---
tags:
- image-classification
- pytorch
- medical
library_name: timm
classes:
- Chickenpox
- Cowpox
- HFMD
- Healthy
- Measles
- Monkeypox
---

# Mpox Detection Model

Ce modèle est un Vision Transformer (ViT) finetuné pour détecter différentes maladies de la peau, notamment la variole du singe (Mpox).

## Classes
Le modèle peut classifier les images dans les catégories suivantes :
- Chickenpox (Varicelle)
- Cowpox (Variole bovine)
- HFMD (Syndrome pieds-mains-bouche)
- Healthy (Sain)
- Measles (Rougeole)
- Monkeypox (Variole du singe)

## Usage

```python
import torch
from models import create_model
from torchvision import transforms
from PIL import Image

# Charger le modèle
model = create_model(pretrained=False)
model.load_state_dict(torch.load("best_mpox_model.pth"))
model.eval()

# Prétraitement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prédiction
img = Image.open("path/to/image.jpg")
img_t = transform(img).unsqueeze(0)
output = model(img_t)
predicted_idx = output.argmax(1).item()
print(predicted_idx)
```
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("Upload des fichiers...")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="best_mpox_model.pth",
            repo_id=REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo="models.py",
            repo_id=REPO_ID,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Upload terminé avec succès !")
        print(f"Votre modèle est disponible ici : https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Erreur lors de l'upload : {e}")

if __name__ == "__main__":
    # Vérification des fichiers
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur: {MODEL_PATH} introuvable.")
    elif not os.path.exists(MODEL_FILE):
        print(f"Erreur: {MODEL_FILE} introuvable.")
    else:
        upload_to_hf()
