from roboflow import Roboflow

rf = Roboflow(api_key="2kGybExbigEojBx3hOJE")  # ta clé API
project = rf.workspace("mpox-jux0x").project("mpox2-n7c6o")  # ton projet
version = project.version(1)  # version du dataset
dataset = version.download("folder") 

# Télécharger le modèle PyTorch
model_path = version.model.download("pt")  # <- utiliser "pt" au lieu de "pytorch"
print("Modèle téléchargé ici :", model_path)
