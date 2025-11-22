---
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
