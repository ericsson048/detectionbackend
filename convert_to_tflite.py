"""
Solution de secours : Conversion en 2 étapes
1. PyTorch → ONNX (déjà fait)
2. Utiliser un service en ligne ou une conversion alternative
"""
import torch
import torch.nn as nn
from torchvision import models
import subprocess
import sys
import os

print("🚀 Conversion PyTorch → TFLite (Méthode Alternative)")

# ===============================
# 1️⃣ Charger et exporter ONNX
# ===============================
print("\n📦 Chargement du modèle PyTorch...")
num_classes = 6
device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
state_dict = torch.load("best_mpox_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("✅ Modèle PyTorch chargé")

print("\n🔄 Export vers ONNX...")
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "skin_disease_model.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("✅ ONNX créé : skin_disease_model.onnx")

# ===============================
# 2️⃣ Tentative de conversion avec onnx2tf
# ===============================
print("\n🔄 Tentative de conversion avec onnx2tf...")
try:
    result = subprocess.run(
        [
            sys.executable, "-m", "onnx2tf",
            "-i", "skin_disease_model.onnx",
            "-o", "skin_disease_tf_model",
            "-osd"  # output_signaturedefs
        ],
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes max
    )
    
    if result.returncode == 0:
        print("✅ Conversion ONNX → TF réussie")
    else:
        print(f"⚠️ Erreur onnx2tf : {result.stderr}")
        raise Exception("Conversion échouée")
        
except Exception as e:
    print(f"❌ Échec de onnx2tf : {e}")
    print("\n" + "="*60)
    print("💡 SOLUTION ALTERNATIVE")
    print("="*60)
    print("\n📋 Utilisez ce service en ligne gratuit :")
    print("   https://convertmodel.com/")
    print("\nÉtapes :")
    print("   1. Uploadez 'skin_disease_model.onnx'")
    print("   2. Choisissez 'Convert to TensorFlow Lite'")
    print("   3. Téléchargez le fichier .tflite")
    print("   4. Renommez-le en 'skin_disease_model.tflite'")
    print("\n📋 OU utilisez Google Colab (gratuit) :")
    print("   https://colab.research.google.com/")
    print("\nCode à exécuter dans Colab :")
    print("""
# Dans Google Colab :
!pip install onnx tf2onnx tensorflow

import tensorflow as tf
import tf2onnx

# Uploadez votre fichier .onnx
spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_onnx_path(
    "skin_disease_model.onnx", 
    input_signature=spec,
    opset=13
)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf.function(model_proto).get_concrete_function(*spec)]
)
tflite_model = converter.convert()

with open("skin_disease_model.tflite", "wb") as f:
    f.write(tflite_model)
    
# Téléchargez le fichier .tflite
from google.colab import files
files.download("skin_disease_model.tflite")
    """)
    sys.exit(1)

# ===============================
# 3️⃣ Conversion TF → TFLite
# ===============================
print("\n🔄 Conversion TF → TFLite...")
try:
    import tensorflow as tf
    
    converter = tf.lite.TFLiteConverter.from_saved_model("skin_disease_tf_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open("skin_disease_model.tflite", "wb") as f:
        f.write(tflite_model)
    
    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ TFLite créé : skin_disease_model.tflite ({file_size_mb:.2f} MB)")
    
    # Test rapide
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    print("✅ Modèle validé !")
    
except Exception as e:
    print(f"❌ Erreur conversion TFLite : {e}")
    sys.exit(1)

print("\n" + "="*60)
print("🎉 CONVERSION RÉUSSIE !")
print("="*60)
print("\n📂 Fichiers créés :")
print("  ├── skin_disease_model.onnx")
print("  ├── skin_disease_tf_model/")
print("  └── skin_disease_model.tflite ✨")
print("\n📱 Prochaine étape : Intégrer dans Flutter")