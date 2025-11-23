import torch
import onnx
import subprocess
import os
import shutil
import tensorflow as tf
import torchvision.models as models

print("\n=========== 🔥 Conversion PyTorch → ONNX → TF → TFLite ===========\n")

# ============================================================
# 1️⃣ CHARGER TON MODÈLE PYTORCH
# ============================================================
print("📦 Chargement du modèle PyTorch...")

# Exemple : ResNet18 (change ici pour ton modèle)
model = models.resnet18(weights=None)
model.eval()

dummy = torch.randn(1, 3, 224, 224)
onnx_path = "model.onnx"

print("✅ Modèle chargé !")


# ============================================================
# 2️⃣ EXPORT PYTORCH → ONNX
# ============================================================
print("\n📜 Export vers ONNX...")

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print(f"✅ ONNX généré : {onnx_path}")


# ============================================================
# 3️⃣ ONNX → TensorFlow SavedModel (via onnx2tf)
# ============================================================
print("\n🔄 Conversion ONNX → TensorFlow SavedModel...")

tf_folder = "tf_model"

# supprimer ancien dossier
if os.path.exists(tf_folder):
    shutil.rmtree(tf_folder)

# exécuter la commande
cmd = f"onnx2tf -i {onnx_path} -o {tf_folder}"
process = subprocess.run(cmd, shell=True)

if process.returncode != 0:
    raise RuntimeError("❌ Échec de onnx2tf. Vérifie onnx2tf est bien installé.")

print("✅ TensorFlow SavedModel généré !")


# ============================================================
# 4️⃣ TensorFlow → TFLite
# ============================================================
print("\n🔄 Conversion TensorFlow → TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(tf_folder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_path = "model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite généré : {tflite_path}")

print("\n🎉 Conversion terminée avec succès !\n")
