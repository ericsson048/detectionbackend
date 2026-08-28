"""Génération de rapports PDF côté serveur avec ReportLab.

Gère l'encodage des caractères accentués en enregistrant une police TTF
Unicode (DejaVuSans si dispo, sinon une police système). Produit un rapport
consolidé (dossier de maladie) ou une seule prédiction.
"""
import io
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

TEAL = colors.HexColor("#00D2B4")
DARK = colors.HexColor("#0F172A")
SUBTLE = colors.HexColor("#64748B")
AMBER = colors.HexColor("#F59E0B")
RED = colors.HexColor("#EF4444")


def _register_font():
    """Enregistre une police TTF Unicode pour gérer les accents."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf"),
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    fonts_registered = set(pdfmetrics.getRegisteredFontNames())
    for path in candidates:
        if os.path.exists(path) and "DejaVuSans" not in fonts_registered:
            try:
                pdfmetrics.registerFont(TTFont("DejaVuSans", path))
                return "DejaVuSans"
            except Exception:
                continue
    return None


_FONT_NAME = None


def _font():
    global _FONT_NAME
    if _FONT_NAME is None:
        reg = _register_font()
        _FONT_NAME = reg or "Helvetica"
    return _FONT_NAME


def _styles():
    f = _font()
    return {
        "title": ParagraphStyle("title", fontName=f, fontSize=22, leading=26, textColor=DARK, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName=f, fontSize=11, leading=14, textColor=SUBTLE),
        "section": ParagraphStyle("section", fontName=f, fontSize=15, leading=18, textColor=DARK, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=f, fontSize=11, leading=16, textColor=DARK),
        "small": ParagraphStyle("small", fontName=f, fontSize=9, leading=12, textColor=SUBTLE),
        "white": ParagraphStyle("white", fontName=f, fontSize=16, leading=20, textColor=colors.white),
        "smallWhite": ParagraphStyle("smallWhite", fontName=f, fontSize=11, leading=14, textColor=colors.white),
    }


def _esc(text: str) -> str:
    """Échappe les caractères spéciaux XML/HTML (ReportLab/Platypus)."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_date(dt) -> str:
    if not dt:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except Exception:
            return dt
    return dt.strftime("%d/%m/%Y à %H:%M")


def build_prediction_pdf(patient_name, prediction, prediction_fr, confidence, advice, notes=None, timestamp=None) -> bytes:
    """Construit un PDF pour une seule prédiction."""
    buf = io.BytesIO()
    st = _styles()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title="Rapport SkinDetect",
    )

    story = []
    story.append(Paragraph("SkinDetect", ParagraphStyle("h", fontName=_font(), fontSize=18, textColor=TEAL)))
    story.append(Paragraph("Rapport Dermatologique Assisté par IA", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Compte-rendu d'Analyse Cutanée", st["title"]))
    story.append(Paragraph("Ce document est généré automatiquement et ne constitue pas un diagnostic médical.", st["subtitle"]))
    story.append(Spacer(1, 16))

    # Infos patient
    story.append(Paragraph("Informations du patient", st["section"]))
    info = Table(
        [[Paragraph("<b>Patient :</b> " + _esc(patient_name), st["body"]),
          Paragraph("<b>Date :</b> " + _esc(_fmt_date(timestamp)), st["body"])]],
        colWidths=[90 * mm, 90 * mm],
    )
    info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("ROUNDEDCORNERS", [10, 10, 10, 10]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(info)
    story.append(Spacer(1, 12))

    # Résultat
    is_healthy = prediction.lower().find("healthy") >= 0
    severity = AMBER
    if is_healthy:
        severity = colors.HexColor("#10B981")
    elif confidence >= 0.8:
        severity = RED

    story.append(Paragraph("Résultat de l'analyse", st["section"]))
    result_box = Table([[Paragraph(_esc(prediction_fr), st["white"]),
                         Paragraph(_esc(f"Confiance : {confidence*100:.1f}%"), st["smallWhite"])]])
    result_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), severity),
        ("ROUNDEDCORNERS", [16, 16, 16, 16]),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(result_box)
    story.append(Spacer(1, 16))

    # Recommandations
    story.append(Paragraph("Recommandations & Protocole", st["section"]))
    rec_lines = [l for l in (advice or "").split("\n") if l.strip()]
    rec_table = Table([[Paragraph(_esc("• " + l), st["body"])] for l in rec_lines] or [[Paragraph("Aucune recommandation disponible.", st["body"])]])
    rec_table.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 4, AMBER),
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.96, 0.94, 0.85)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(rec_table)

    if notes:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Notes de suivi", st["section"]))
        notes_table = Table([[Paragraph(_esc(notes), st["body"])]])
        notes_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9"))]))
        story.append(notes_table)

    story.append(Spacer(1, 18))
    story.append(Paragraph("Avertissement légal", st["section"]))
    warn = Table([[Paragraph(
        "Ce rapport est produit par un modèle d'intelligence artificielle à des fins éducatives. "
        "Il ne remplace pas une consultation avec un professionnel de santé qualifié. En cas de "
        "symptômes graves (fièvre élevée, difficultés respiratoires, confusion), consultez immédiatement un médecin.",
        st["body"])]])
    warn.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 4, RED),
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.98, 0.9, 0.9)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(warn)
    story.append(Spacer(1, 16))
    story.append(Paragraph(f"Généré le {_fmt_date(datetime.utcnow())} par SkinDetect AI", st["small"]))

    doc.build(story)
    return buf.getvalue()


def build_casefile_pdf(patient_name, case_file, predictions) -> bytes:
    """Construit un PDF consolidé d'un dossier de maladie (historique d'évolution)."""
    buf = io.BytesIO()
    st = _styles()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=24 * mm, rightMargin=24 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title="Dossier SkinDetect",
    )

    story = []
    story.append(Paragraph("SkinDetect", ParagraphStyle("h", fontName=_font(), fontSize=18, textColor=TEAL)))
    story.append(Paragraph("Dossier de suivi - Rapport Dermatologique", st["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL))
    story.append(Spacer(1, 8))
    story.append(Paragraph(_esc(case_file.title or "Dossier de maladie"), st["title"]))
    story.append(Paragraph(_esc(f"Statut : {case_file.status}") + ". Document non médical, généré automatiquement.", st["subtitle"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Patient : {_esc(patient_name)}", st["body"]))
    story.append(Paragraph(f"Nombre d'examens : {len(predictions)}", st["body"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Historique des examens", st["section"]))

    sorted_preds = sorted(predictions, key=lambda p: p.timestamp or datetime.min)
    for i, p in enumerate(sorted_preds, 1):
        is_healthy = (p.prediction or "").lower().find("healthy") >= 0
        color = colors.HexColor("#10B981") if is_healthy else (RED if p.confidence >= 0.8 else AMBER)
        box = Table([[Paragraph(f"<b>Examen n°{i}</b> — {_esc(p.prediction or '')}   ({p.confidence*100:.1f}%)   "
                                 f"{_esc(_fmt_date(p.timestamp))}", st["body"])]])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), color),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("ROUNDEDCORNERS", [10, 10, 10, 10]),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(box)
        story.append(Spacer(1, 6))
        advice_lines = [l for l in (p.advice or "").split("\n") if l.strip()]
        for l in advice_lines[:6]:
            story.append(Paragraph(_esc("• " + l), st["small"]))
        if p.notes:
            story.append(Paragraph(_esc("Notes : " + p.notes), st["small"]))
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Avertissement légal", st["section"]))
    story.append(Paragraph(
        "Ce dossier est produit par une IA à des fins éducatives et ne remplace pas un avis médical.",
        st["body"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Généré le {_fmt_date(datetime.utcnow())} par SkinDetect AI", st["small"]))

    doc.build(story)
    return buf.getvalue()
