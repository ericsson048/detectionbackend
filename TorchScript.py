import torch
import torch.nn as nn
import torchvision.models as models

# ===============================
# 1️⃣ Définir l'architecture exacte
# ===============================
# Exemple : ResNet18 avec 6 classes
num_classes = 6  # adapte selon ton modèle
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ===============================
# 2️⃣ Charger les poids .pth
# ===============================
state_dict = torch.load("best_mpox_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# ===============================
# 3️⃣ Mettre en mode évaluation
# ===============================
model.eval()

# ===============================
# 4️⃣ Créer un exemple d'entrée
# ===============================
example_input = torch.randn(1, 3, 224, 224)  # 1 image, 3 canaux, 224x224

# ===============================
# 5️⃣ Tracer le modèle en TorchScript
# ===============================
traced_script_module = torch.jit.trace(model, example_input)

# ===============================
# 6️⃣ Sauvegarder le modèle TorchScript
# ===============================
traced_script_module.save("detection.pt")

print("✅ Conversion terminée !")
print("Tu as maintenant :")
print("- best_mpox_model.pth (original)")
print("- detection.pt (mobile-friendly pour React Native)")
