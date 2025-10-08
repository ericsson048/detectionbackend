# 🔹 Installer Roboflow si nécessaire
# pip install roboflow

from roboflow import Roboflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# ===============================
# 1️⃣ CONFIGURATION
# ===============================
API_KEY = "2kGybExbigEojBx3hOJE"
WORKSPACE = "mpox-jux0x"
PROJECT   = "mpox2-n7c6o"
VERSION   = 2
BATCH_SIZE = 16
EPOCHS = 50
MODEL_SAVE_PATH = "best_mpox_model.pth"


# ===============================
# 2️⃣ Connexion Roboflow et téléchargement dataset
# ===============================
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("folder")  # format pour récupérer les images

train_dir = os.path.join(dataset.location, "train")
valid_dir = os.path.join(dataset.location, "valid")
test_dir  = os.path.join(dataset.location, "test")


def list_classes(path):
    if not os.path.exists(path):
        return []
    # Liste uniquement les dossiers (chaque dossier = une classe)
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

# 🔍 Lister les classes de chaque split
train_classes = list_classes(train_dir)
valid_classes = list_classes(valid_dir)
test_classes  = list_classes(test_dir)

print("📊 Classes trouvées dans chaque split :")
print(f"Train ({len(train_classes)} classes) :", train_classes)
print(f"Valid ({len(valid_classes)} classes) :", valid_classes)
print(f"Test  ({len(test_classes)} classes) :", test_classes)

# ✅ Vérifier si toutes les classes sont identiques entre splits
if train_classes == valid_classes == test_classes:
    print("\n✅ Tous les splits ont les mêmes classes, dataset OK.")
else:
    print("\n⚠️ Attention : les splits n’ont pas les mêmes classes !")
    print("Différences :")
    print("Classes manquantes dans VALID :", set(train_classes) - set(valid_classes))
    print("Classes manquantes dans TEST  :", set(train_classes) - set(test_classes))
    print("Classes en trop dans VALID    :", set(valid_classes) - set(train_classes))
    print("Classes en trop dans TEST     :", set(test_classes) - set(train_classes))


# ===============================
# 3️⃣ Transformations
# ===============================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 4️⃣ Datasets et DataLoaders
# ===============================
train_dataset = ImageFolder(train_dir, transform=train_transforms)
valid_dataset = ImageFolder(valid_dir, transform=valid_transforms)
test_dataset  = ImageFolder(test_dir, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ===============================
# 5️⃣ Modèle ResNet18 pré-entraîné
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===============================
# 6️⃣ Fonction d'entraînement avec sauvegarde du meilleur modèle
# ===============================
def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10, save_path="best_model.pth"):
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Sauvegarder le meilleur modèle
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with accuracy: {best_acc:.4f}")

# ===============================
# 7️⃣ Lancer l'entraînement
# ===============================
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=EPOCHS, save_path=MODEL_SAVE_PATH)

# ===============================
# 8️⃣ Tester le modèle
# ===============================
def test_model(model, test_loader, model_path=MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"🎯 Test Accuracy: {correct/total:.4f}")

test_model(model, test_loader)
