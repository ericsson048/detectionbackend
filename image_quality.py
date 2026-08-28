"""Détection de la qualité d'image côté serveur (backend).

Repli de la logique du contrôle qualité photo du client Flutter :
- Luminosité moyenne (luma pondérée 0.299/0.587/0.114).
- Netteté estimée par la variance du Laplacien sur l'image en niveaux de gris.

Utilise uniquement Pillow (aucun besoin d'OpenCV).
"""
import io
from PIL import Image, ImageFilter, ImageOps


def analyze_quality(image_bytes: bytes) -> dict:
    """Analyse la qualité d'une image et retourne un dict compatible schemas.ImageQualityResponse."""
    issues = []
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {
            "usable": False,
            "average_brightness": 0.0,
            "sharpness_variance": 0.0,
            "issues": [{
                "code": "decode_error",
                "title": "Image illisible",
                "description": "Impossible de décoder l'image. Le format n'est pas pris en charge.",
                "suggestion": "Utilisez une photo JPG ou PNG valide (camera ou galerie).",
            }],
        }

    avg_brightness = _average_brightness(image)
    sharpness = _laplacian_variance(image)

    if avg_brightness < 45.0:
        issues.append({
            "code": "too_dark",
            "title": "Photo trop sombre",
            "description": "La luminosité moyenne est très faible, les détails de la lésion peuvent être masqués.",
            "suggestion": "Rapprochez-vous d'une source de lumière ou activez le flash.",
        })
    elif avg_brightness > 235.0:
        issues.append({
            "code": "overexposed",
            "title": "Photo trop claire / surexposée",
            "description": "L'image est surexposée, les contrastes de la peau sont perdus.",
            "suggestion": "Éloignez la lumière directe ou ajustez l'exposition de la caméra.",
        })

    if sharpness < 100.0:
        issues.append({
            "code": "blurry",
            "title": "Photo floue",
            "description": "La netteté est insuffisante pour une analyse fiable des détails cutanés.",
            "suggestion": "Stabilisez le téléphone, restez immobile et cadrez à ~15 cm.",
        })

    return {
        "usable": len(issues) == 0,
        "average_brightness": round(avg_brightness, 2),
        "sharpness_variance": round(sharpness, 2),
        "issues": issues,
    }


def _average_brightness(image: Image.Image) -> float:
    """Luminosité moyenne (0-255) calculée sur les canaux RGB, échantillonnée."""
    # Redimensionner pour la performance sur les grandes images
    thumb = image.copy()
    thumb.thumbnail((512, 512))
    gray = ImageOps.grayscale(thumb)
    pixels = list(gray.getdata())
    if not pixels:
        return 128.0
    return sum(pixels) / len(pixels)


def _laplacian_variance(image: Image.Image) -> float:
    """Netteté estimée par la variance du Laplacien sur l'image en niveaux de gris."""
    small = image.copy()
    small.thumbnail((160, 160))
    gray = ImageOps.grayscale(small)

    # Laplacian approximé par le filtre sharp de Pillow retournant la différence
    laplacian = gray.filter(ImageFilter.Kernel((3, 3), (
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    ), scale=1))

    data = list(laplacian.getdata())
    # Moyenne
    mean = sum(data) / len(data)
    variance = sum((p - mean) ** 2 for p in data) / len(data)
    return variance
