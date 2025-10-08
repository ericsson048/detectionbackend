import timm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import io
import os

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1  # Pour test rapide
MODEL_PATH = "mpox_vit_model_local.pth"
DATASET_PATH = "Mpox2-1"

# --- TRANSFORMATIONS ---
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- DATASET LOCAL ---
train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=transform_train)

if os.path.exists(os.path.join(DATASET_PATH, "valid")):
    val_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "valid"), transform=transform_val)
else:
    val_dataset = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) if val_dataset else None

classes = train_dataset.classes
num_classes = len(classes)

# --- MODELE ---
def create_model(pretrained=True):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

# --- CHARGEMENT DU MODELE ---
def load_or_train_model():
    need_train = True
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) > 1000:  # Vérifie que le fichier n'est pas vide
            try:
                model = create_model(pretrained=False)
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                model.eval()
                model.to(device)
                print(f"Modèle chargé depuis {MODEL_PATH}")
                need_train = False
                return model
            except Exception as e:
                print(f"Erreur au chargement du modèle: {e}")

    # Entraînement si le modèle n'existe pas ou est corrompu
    print("Entraînement d'un nouveau modèle...")
    model = create_model(pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    model.eval()
    return model

# model = load_or_train_model()

# --- PRÉDICTION ---
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]


