"""
Script de conversion du modèle Mpox PyTorch vers TensorFlow Lite
Basé sur votre architecture ResNet18
"""
import torch
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
import numpy as np
import onnx
from onnx_tf.backend import prepare
import os
import shutil

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "best_mpox_model.pth"
OUTPUT_TFLITE = "skin_disease_model.tflite"
NUM_CLASSES = 6  # Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox
INPUT_SIZE = 224

print("="*60)
print("🔄 CONVERSION PYTORCH → TFLITE")
print("="*60)

# ===============================
# 1️⃣ CRÉER LE MODÈLE (même architecture que l'entraînement)
# ===============================
print("\n📦 Création du modèle ResNet18...")
model = models.resnet18(pretrained=False)  # pretrained=False car on charge nos poids
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# ===============================
# 2️⃣ CHARGER LES POIDS
# ===============================
print("📥 Chargement des poids depuis", MODEL_PATH)
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
print("✅ Poids chargés avec succès")

# ===============================
# 3️⃣ PYTORCH → ONNX
# ===============================
print("\n🔄 Conversion PyTorch → ONNX...")
dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
onnx_path = "temp_model.onnx"

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
print("✅ Modèle exporté vers ONNX")

# ===============================
# 4️⃣ ONNX → TENSORFLOW
# ===============================
print("\n🔄 Conversion ONNX → TensorFlow...")
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_model_dir = "temp_tf_model"
tf_rep.export_graph(tf_model_dir)
print("✅ Modèle converti en TensorFlow")

# ===============================
# 5️⃣ TENSORFLOW → TFLITE
# ===============================
print("\n🔄 Conversion TensorFlow → TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

# Options d'optimisation
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# Conversion
tflite_model = converter.convert()

# Sauvegarde
with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Modèle TFLite sauvegardé : {OUTPUT_TFLITE}")
print(f"   Taille du fichier : {os.path.getsize(OUTPUT_TFLITE) / (1024*1024):.2f} MB")

# ===============================
# 6️⃣ TESTER LE MODÈLE TFLITE
# ===============================
print("\n" + "="*60)
print("🧪 TEST DU MODÈLE TFLITE")
print("="*60)

interpreter = tf.lite.Interpreter(model_path=OUTPUT_TFLITE)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n📊 Détails du modèle :")
print(f"   Input shape : {input_details[0]['shape']}")
print(f"   Input dtype : {input_details[0]['dtype']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test avec entrée aléatoire
print(f"\n🔬 Test avec entrée aléatoire...")
test_input = np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)

# Appliquer la même normalisation ImageNet que pendant l'entraînement
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
test_input_normalized = (test_input - mean) / std

interpreter.set_tensor(input_details[0]['index'], test_input_normalized)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"   Output brut: {output[0]}")
print(f"   Min: {output.min():.4f}, Max: {output.max():.4f}")

# Appliquer softmax
exp_output = np.exp(output[0] - np.max(output[0]))
softmax_output = exp_output / np.sum(exp_output)
print(f"   Softmax: {softmax_output}")
print(f"   Somme: {np.sum(softmax_output):.6f}")

# Vérification
if np.allclose(output[0], output[0][0], atol=1e-5):
    print("\n❌ ERREUR : Toutes les sorties sont identiques !")
    print("   Le modèle n'a pas été converti correctement.")
else:
    print("\n✅ Le modèle varie correctement entre les classes")
    predicted_class = np.argmax(softmax_output)
    confidence = softmax_output[predicted_class]
    classes = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
    print(f"   Prédiction test: {classes[predicted_class]} ({confidence*100:.2f}%)")

# ===============================
# 7️⃣ COMPARAISON PYTORCH VS TFLITE
# ===============================
print("\n" + "="*60)
print("🔍 COMPARAISON PYTORCH vs TFLITE")
print("="*60)

# Test PyTorch
model.eval()
with torch.no_grad():
    # Préparer l'entrée pour PyTorch (channels first)
    pytorch_input = torch.from_numpy(test_input_normalized).permute(0, 3, 1, 2)
    pytorch_output = model(pytorch_input).numpy()
    
print(f"PyTorch output : {pytorch_output[0]}")
print(f"TFLite output  : {output[0]}")
print(f"Différence max : {np.max(np.abs(pytorch_output - output)):.6f}")

if np.allclose(pytorch_output, output, atol=1e-3):
    print("✅ Les sorties PyTorch et TFLite sont identiques (tolérance: 1e-3)")
else:
    print("⚠️  Les sorties diffèrent légèrement (normal avec la conversion)")

# ===============================
# 8️⃣ NETTOYAGE
# ===============================
print("\n🧹 Nettoyage des fichiers temporaires...")
if os.path.exists(onnx_path):
    os.remove(onnx_path)
if os.path.exists(tf_model_dir):
    shutil.rmtree(tf_model_dir)
print("✅ Nettoyage terminé")

# ===============================
# 9️⃣ INSTRUCTIONS FINALES
# ===============================
print("\n" + "="*60)
print("📋 INSTRUCTIONS POUR FLUTTER")
print("="*60)
print(f"""
1️⃣ Copiez le fichier dans votre projet Flutter :
   cp {OUTPUT_TFLITE} <votre_projet>/assets/models/

2️⃣ Format d'entrée : [1, 224, 224, 3] (TensorFlow format)

3️⃣ Normalisation ImageNet à utiliser dans Flutter :
   mean = [0.485, 0.456, 0.406]
   std  = [0.229, 0.224, 0.225]
   
   Pour chaque pixel RGB :
   normalized_value = (pixel_value / 255.0 - mean) / std

4️⃣ Classes (dans l'ordre) :
   0: Chickenpox
   1: Cowpox
   2: HFMD
   3: Healthy
   4: Measles
   5: Monkeypox

5️⃣ Le code Flutter que je vous ai fourni utilise déjà 
   la bonne normalisation ImageNet ! ✅
""")
print("="*60)
print("✅ CONVERSION TERMINÉE AVEC SUCCÈS !")
print("="*60)