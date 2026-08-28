from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
from PIL import Image
import io
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import jwt
import datetime
import time
from functools import wraps

# ===== IMPORTS EMAIL =====
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr

# ===== IMPORTS SANS PyTorch =====
import database, db_models, schemas, crud, auth
from db_models import LoginRequest, AuthResponse

from ai_provider import AIProvider
from rag import build_context
import image_quality as imgq
import report_pdf
import base64

GEMMA_API_URL = os.getenv("GEMMA_API_URL", "http://localhost:8001")

# Provider IA unifié : Gemma d'abord, AIHubMix en secours.
ai_provider = AIProvider(GEMMA_API_URL)

# ===== CONFIGURATION =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SECRET_KEY = "dantylanez et ericsson"
ALGORITHM = "HS256"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠️ GEMINI_API_KEY non configurée. Conseils IA désactivés.")

# ===== CONFIGURATION EMAIL =====
email_config = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME", "edulms048@gmail.com"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", "ifgq wumr wapv ltmj"),
    MAIL_FROM=os.getenv("MAIL_FROM", "edulms048@gmail.com"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

fm = FastMail(email_config)

# ===== INITIALISATION =====
db_models.Base.metadata.create_all(bind=database.engine)
app = FastAPI(
    title="API Détection de maladies cutanées - Version Légère",
    description="""
    API backend pour l'application de détection de maladies cutanées.
    
    Fonctionnalités principales :
    * **Authentification** : Gestion des utilisateurs (inscription, connexion, JWT).
    * **Prédiction** : Réception des résultats de l'IA mobile et génération de conseils via Gemini.
    * **Historique** : Suivi des prédictions passées.
    * **Notifications** : Envoi d'emails automatiques.
    
    Note : L'inférence ML principale est effectuée côté client (Flutter) avec TFLite.
    """,
    version="1.0.0"
)
get_db = database.get_db

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== FONCTIONS UTILITAIRES =====
def clean_markdown_for_mobile(text: str) -> str:
    """Convertit le markdown en texte formaté pour mobile"""
    text = re.sub(r'#{1,6}\s+(.+)', r'📌 \1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'• \1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^\s*[\*\-]\s+', '  • ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_token(user):
    """Génère un JWT pour l'utilisateur"""
    payload = {
        "userid": user.id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ===== TEMPLATES EMAIL =====
def get_prediction_email_template(username: str, prediction: str, confidence: float, advice: str) -> str:
    """Template HTML pour email de résultat"""
    # Déterminer la couleur selon la gravité
    color = "#dc3545" if prediction.lower() in ["measles", "monkeypox"] else "#667eea"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .container {{
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, {color} 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .content {{
                padding: 30px;
            }}
            .result-box {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid {color};
            }}
            .result-box h2 {{
                margin-top: 0;
                color: {color};
                font-size: 20px;
            }}
            .confidence {{
                display: inline-block;
                background: {color};
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            .advice {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
                white-space: pre-line;
            }}
            .warning {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
                color: #721c24;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                color: #666;
                font-size: 12px;
            }}
            .button {{
                display: inline-block;
                background: {color};
                color: white;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Résultat de votre analyse cutanée</h1>
            </div>
            
            <div class="content">
                <p>Bonjour <strong>{username}</strong>,</p>
                
                <p>Votre analyse a été effectuée avec succès. Voici les résultats :</p>
                
                <div class="result-box">
                    <h2>📊 Diagnostic détecté</h2>
                    <h3 style="color: #333; margin: 10px 0;">{prediction}</h3>
                    <p style="margin-top: 15px;">
                        Niveau de confiance : <span class="confidence">{confidence:.1%}</span>
                    </p>
                </div>
                
                <div class="advice">
                    <strong>💡 Conseils personnalisés :</strong><br><br>
                    {advice}
                </div>
                
                <div class="warning">
                    <strong>⚠️ Avertissement médical :</strong><br>
                    Ces résultats sont générés par intelligence artificielle et ne constituent pas un diagnostic médical officiel. 
                    Consultez toujours un professionnel de santé qualifié pour obtenir un avis médical personnalisé et un traitement approprié.
                </div>
                
                <center>
                    <p style="margin-top: 30px;">
                        <a href="https://votre-app.com" class="button">
                            📱 Ouvrir l'application
                        </a>
                    </p>
                </center>
                
                <p style="color: #666; font-size: 14px; margin-top: 30px;">
                    💡 <strong>Conseil :</strong> Conservez cet email pour votre suivi médical et montrez-le à votre médecin si nécessaire.
                </p>
            </div>
            
            <div class="footer">
                <p><strong>SkinDetect</strong> - Détection de maladies cutanées par IA</p>
                <p>Cet email a été envoyé automatiquement suite à votre analyse.</p>
                <p>Pour toute question : edulms048@gmail.com</p>
            </div>
        </div>
    </body>
    </html>
    """

def get_welcome_email_template(username: str) -> str:
    """Template HTML pour email de bienvenue"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .container {{
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .content {{
                padding: 30px;
            }}
            .feature-box {{
                background: #f8f9fa;
                padding: 15px;
                margin: 15px 0;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }}
            .feature-box strong {{
                color: #667eea;
                font-size: 16px;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>👋 Bienvenue sur SkinDetect !</h1>
                <p style="margin: 10px 0 0 0; font-size: 16px;">Votre assistant santé cutanée par IA</p>
            </div>
            
            <div class="content">
                <p>Bonjour <strong>{username}</strong>,</p>
                
                <p>Merci de vous être inscrit sur <strong>SkinDetect</strong> ! Nous sommes ravis de vous accompagner dans le suivi de votre santé cutanée.</p>
                
                <h3 style="color: #667eea;">🎯 Ce que vous pouvez faire :</h3>
                
                <div class="feature-box">
                    <strong>📸 Analyse instantanée</strong><br>
                    Prenez une photo de votre peau et obtenez un diagnostic préliminaire en quelques secondes grâce à notre IA avancée.
                </div>
                
                <div class="feature-box">
                    <strong>💡 Conseils personnalisés</strong><br>
                    Recevez des recommandations adaptées à votre condition détectée, avec des conseils pratiques et des mesures préventives.
                </div>
                
                <div class="feature-box">
                    <strong>📊 Historique complet</strong><br>
                    Suivez l'évolution de vos analyses dans le temps et conservez un historique détaillé de vos résultats.
                </div>
                
                <div class="feature-box">
                    <strong>📧 Notifications email</strong><br>
                    Recevez vos résultats directement par email pour les partager facilement avec votre médecin.
                </div>
                
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107;">
                    <strong>⚠️ Rappel important :</strong><br>
                    SkinDetect est un outil d'aide à la décision, <strong>pas un substitut</strong> à une consultation médicale professionnelle. 
                    Consultez toujours un dermatologue pour un diagnostic confirmé.
                </div>
                
                <center style="margin-top: 30px;">
                    <p><strong>Prêt à commencer ?</strong></p>
                    <p style="color: #666;">Ouvrez l'application mobile et effectuez votre première analyse !</p>
                </center>
            </div>
            
            <div class="footer">
                <p><strong>SkinDetect</strong> - Détection de maladies cutanées par IA</p>
                <p>Des questions ? Contactez-nous : edulms048@gmail.com</p>
                <p style="margin-top: 10px; color: #999;">
                    Vous recevez cet email car vous vous êtes inscrit sur SkinDetect.
                </p>
            </div>
        </div>
    </body>
    </html>
    """

# ===== FONCTIONS D'ENVOI EMAIL =====
async def send_prediction_email(
    user_email: str,
    username: str,
    prediction: str,
    confidence: float,
    advice: str
):
    """Envoie un email avec les résultats de prédiction"""
    try:
        html = get_prediction_email_template(username, prediction, confidence, advice)
        
        message = MessageSchema(
            subject=f"✅ Résultat d'analyse : {prediction}",
            recipients=[user_email],
            body=html,
            subtype=MessageType.html
        )
        
        await fm.send_message(message)
        print(f"✅ Email de résultat envoyé à {user_email}")
        
    except Exception as e:
        print(f"❌ Erreur envoi email de résultat : {e}")

async def send_welcome_email(user_email: str, username: str):
    """Envoie un email de bienvenue"""
    try:
        html = get_welcome_email_template(username)
        
        message = MessageSchema(
            subject="👋 Bienvenue sur SkinDetect - Votre assistant santé cutanée",
            recipients=[user_email],
            body=html,
            subtype=MessageType.html
        )
        
        await fm.send_message(message)
        print(f"✅ Email de bienvenue envoyé à {user_email}")
        
    except Exception as e:
        print(f"❌ Erreur envoi email de bienvenue : {e}")

def get_otp_email_template(username: str, otp_code: str) -> str:
    """Template HTML pour l'envoi du code OTP"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .container {{
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .content {{
                padding: 30px;
            }}
            .otp-box {{
                background: #f8f9fa;
                border: 2px dashed #667eea;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                margin: 20px 0;
            }}
            .otp-code {{
                font-size: 36px;
                font-weight: bold;
                letter-spacing: 8px;
                color: #667eea;
            }}
            .warning {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
                color: #721c24;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                background: #f8f9fa;
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 Réinitialisation de mot de passe</h1>
            </div>
            <div class="content">
                <p>Bonjour <strong>{username}</strong>,</p>
                <p>Vous avez demandé la réinitialisation de votre mot de passe SkinDetect.</p>
                
                <div class="otp-box">
                    <p style="margin: 0;">Votre code de vérification :</p>
                    <p class="otp-code">{otp_code}</p>
                    <p style="margin: 0; color: #666; font-size: 13px;">Ce code expire dans 15 minutes.</p>
                </div>
                
                <div class="warning">
                    <strong>⚠️ Sécurité :</strong> Si vous n'êtes pas à l'origine de cette demande, ignorez cet email
                    et ne partagez ce code avec personne.
                </div>
            </div>
            <div class="footer">
                <p><strong>SkinDetect</strong> - Détection de maladies cutanées par IA</p>
            </div>
        </div>
    </body>
    </html>
    """

async def send_otp_email(user_email: str, username: str, otp_code: str):
    """Envoie un email contenant le code OTP"""
    try:
        html = get_otp_email_template(username, otp_code)
        message = MessageSchema(
            subject="🔐 Votre code de réinitialisation SkinDetect",
            recipients=[user_email],
            body=html,
            subtype=MessageType.html
        )
        await fm.send_message(message)
        print(f"✅ Email OTP envoyé à {user_email}")
    except Exception as e:
        print(f"❌ Erreur envoi email OTP : {e}")
        raise

# ===== EMAIL DE RAPPORT PDF =====
async def send_report_email(user_email: str, username: str, pdf_bytes: bytes):
    """Envoie un rapport PDF par email au patient / dermatologue."""
    try:
        message = MessageSchema(
            subject="📄 Votre rapport SkinDetect",
            recipients=[user_email],
            body=(
                "<p>Bonjour <strong>{}</strong>,</p>"
                "<p>Veuillez trouver ci-joint votre rapport dermatologique généré par SkinDetect.</p>"
                "<p>Ce rapport est produit par une IA à des fins éducatives et ne remplace pas un avis médical.</p>"
            ).format(username),
            subtype=MessageType.html,
            attachments=[
                {
                    "file": "rapport_skindetect.pdf",
                    "data": pdf_bytes,
                    "mimetype": "application/pdf",
                }
            ],
        )
        await fm.send_message(message)
        print(f"✅ Rapport PDF envoyé à {user_email}")
    except Exception as e:
        print(f"❌ Erreur envoi rapport PDF : {e}")

# ===== RAPPELS EMAIL (expéditeur) =====
async def send_reminder_email(user_email: str, title: str, message: str | None):
    """Envoie un email de rappel programmé."""
    try:
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
            "<div style='font-family:Arial;max-width:600px;margin:auto;padding:20px'>"
            "<h2 style='color:#00D2B4'>🔔 Rappel SkinDetect</h2>"
            f"<h3 style='color:#0F172A'>{title}</h3>"
            f"<p style='color:#333'>{message or ''}</p>"
            "<p style='color:#888;font-size:12px'>Ce rappel est envoyé automatiquement par SkinDetect.</p>"
            "</div></body></html>"
        )
        message = MessageSchema(
            subject=f"🔔 Rappel : {title}",
            recipients=[user_email],
            body=html,
            subtype=MessageType.html,
        )
        await fm.send_message(message)
        print(f"✅ Rappel envoyé à {user_email} : {title}")
    except Exception as e:
        print(f"❌ Erreur envoi rappel email : {e}")

# ===== DICTIONNAIRE DE CONSEILS STATIQUES =====
SYSTEM_PROMPT = """
Tu es un assistant médical éducatif spécialisé en dermatologie.
Tu n’es pas un médecin.
Tu n’établis aucun diagnostic médical.
Tu expliques les résultats fournis par un modèle d’analyse d’image.
Tu utilises un langage simple, rassurant et responsable.
Tu indiques clairement quand consulter un professionnel de santé.
Tu adaptes les conseils à des contextes à faibles ressources.
Tu ne prescris jamais de médicaments.
"""

STATIC_ADVICE = {
    "chickenpox": {
        "description": "La varicelle est une infection virale très contagieuse causant des éruptions de vésicules.",
        "recommendations": [
            "Isolez-vous pour éviter la contagion (surtout femmes enceintes et bébés)",
            "Ne grattez pas les boutons pour éviter les cicatrices et infections",
            "Appliquez des compresses fraîches et des crèmes apaisantes",
            "Coupez vos ongles courts et portez des gants la nuit si nécessaire"
        ],
        "warning": "⚠️ Consultez un médecin immédiatement si : fièvre élevée, difficultés respiratoires, ou si vous êtes immunodéprimé."
    },
    "cowpox": {
        "description": "La vaccine (cowpox) est une infection virale rare transmise par contact avec des animaux infectés.",
        "recommendations": [
            "Évitez de toucher la lésion et lavez-vous soigneusement les mains",
            "Couvrez la zone infectée avec un pansement propre",
            "Ne partagez pas vos objets personnels (serviettes, vêtements)",
            "Surveillez les signes d'infection secondaire (pus, rougeur croissante)"
        ],
        "warning": "Consultez un médecin pour confirmer le diagnostic et écarter d'autres infections virales plus graves."
    },
    "hfmd": {
        "description": "Le syndrome pieds-mains-bouche (HFMD) est une infection virale courante chez les enfants.",
        "recommendations": [
            "Hydratez-vous bien et mangez des aliments mous si la bouche est douloureuse",
            "Lavez-vous fréquemment les mains, surtout après contact avec les lésions",
            "Évitez le contact rapproché avec d'autres personnes pendant 7-10 jours",
            "Désinfectez les surfaces et jouets régulièrement"
        ],
        "warning": "Consultez un médecin si : forte fièvre, déshydratation, maux de tête sévères ou raideur de la nuque."
    },
    "healthy": {
        "description": "Aucune anomalie cutanée détectée. Votre peau semble en bonne santé !",
        "recommendations": [
            "Continuez à protéger votre peau du soleil (SPF 30+ minimum)",
            "Maintenez une bonne hydratation quotidienne",
            "Adoptez une alimentation équilibrée riche en vitamines",
            "Surveillez tout changement inhabituel (grains de beauté, taches)"
        ],
        "warning": "Consultez un dermatologue une fois par an pour un examen préventif, surtout si antécédents familiaux."
    },
    "measles": {
        "description": "La rougeole est une infection virale très contagieuse et potentiellement grave.",
        "recommendations": [
            "⚠️ ISOLATION STRICTE : Restez chez vous pendant au moins 4 jours après l'éruption",
            "Reposez-vous dans une pièce sombre (sensibilité à la lumière)",
            "Hydratez-vous abondamment et prenez du paracétamol pour la fièvre",
            "Évitez tout contact avec personnes non vaccinées, femmes enceintes et bébés"
        ],
        "warning": "⚠️ URGENCE MÉDICALE : Consultez immédiatement si complications (difficultés respiratoires, convulsions, confusion). La rougeole peut être mortelle."
    },
    "monkeypox": {
        "description": "La variole du singe (Mpox) est une infection virale avec éruption cutanée caractéristique.",
        "recommendations": [
            "⚠️ ISOLEMENT REQUIS : Évitez tout contact physique jusqu'à guérison complète",
            "Couvrez les lésions avec des pansements et changez-les régulièrement",
            "Désinfectez toutes les surfaces touchées et lavez le linge à haute température",
            "Ne partagez aucun objet personnel (literie, vêtements, ustensiles)"
        ],
        "warning": "⚠️ DÉCLARATION OBLIGATOIRE : Contactez immédiatement votre médecin ou les autorités sanitaires. Cette maladie nécessite un suivi médical strict."
    },
    "default": {
        "description": "Affection cutanée détectée nécessitant une évaluation professionnelle.",
        "recommendations": [
            "Maintenez une bonne hygiène de la peau",
            "Évitez de gratter ou irriter la zone affectée",
            "Documentez l'évolution avec des photos datées",
            "Notez tout symptôme associé (fièvre, douleur, démangeaisons)"
        ],
        "warning": "Seul un professionnel de santé peut établir un diagnostic précis. Consultez rapidement."
    }
}

def get_static_advice(prediction: str) -> str:
    """
    Génère des conseils statiques formatés basés sur la prédiction.
    Recherche par correspondance partielle (insensible à la casse).
    """
    prediction_lower = prediction.lower()
    
    # Recherche de correspondance
    advice_data = None
    for key in STATIC_ADVICE.keys():
        if key in prediction_lower or prediction_lower in key:
            advice_data = STATIC_ADVICE[key]
            break
    
    # Si aucune correspondance, utiliser les conseils par défaut
    if not advice_data:
        advice_data = STATIC_ADVICE["default"]
    
    # Formatage pour mobile
    formatted_advice = f"📌 {advice_data['description']}\n\n"
    formatted_advice += "💡 Recommandations :\n"
    for i, rec in enumerate(advice_data['recommendations'], 1):
        formatted_advice += f"  • {rec}\n"
    formatted_advice += f"\n⚠️ {advice_data['warning']}"
    
    return formatted_advice


def generate_gemma_advice(prediction: str, confidence: float) -> str:
    # ===== RAG : récupérer un contexte fiable lié à la prédiction =====
    rag_context = build_context(prediction, top_k=2)

    context_block = ""
    if rag_context:
        context_block = f"""
CONTEXTE MÉDICAL DE RÉFÉRENCE (généralement fiable, à reformuler avec tes propres mots) :
{rag_context}

"""

    prompt = f"""{context_block}Un modèle d'analyse d'image cutanée suggère :
- Affection possible : {prediction}
- Probabilité : {confidence*100:.1f} %

En t'appuyant sur le contexte fourni si disponible, explique :
1. Ce que cela signifie
2. Les gestes simples à faire
3. Quand consulter un médecin
4. Un message rassurant
"""

    response = ai_provider.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.4,
        max_tokens=350
    )

    return response

def get_ai_advice_with_fallback(prediction: str, confidence: float) -> str:
    try:
        if confidence < 0.6:
            return get_static_advice(prediction)

        ai_text = generate_gemma_advice(prediction, confidence)

        if not ai_text or len(ai_text) < 50:
            return get_static_advice(prediction)

        return clean_markdown_for_mobile(ai_text)

    except Exception as e:
        print("⚠️ Gemma indisponible :", e)
        return get_static_advice(prediction)


# ===== PROMPS CHATBOT MÉDICAL =====
CHAT_SYSTEM_PROMPT = """
Tu es "SkinDetect Care", un assistant médical éducatif spécialisé en dermatologie.
Tu n'es PAS un médecin et tu n'établis AUCUN diagnostic médical.
Lignes directrices strictes :
1. Réponds uniquement en français, de manière simple, claire et rassurante.
2. Ne prescris JAMAIS de médicaments, doses ou traitements.
3. Explique les conditions cutanées courantes (varicelle, rougeole, mpox, pieds-mains-bouche, etc.) de façon éducative.
4. Donne des mesures préventives et d'hygiène générales.
5. Indique clairement quand consulter un professionnel de santé (fièvre élevée, difficultés respiratoires, symptômes graves).
6. Toujours terminer par un rappel que ce n'est pas un diagnostic médical et que le patient doit consulter un médecin si inquiet.
7. Adapte tes conseils aux contextes à faibles ressources.
"""

QUICK_CHAT_SUGGESTIONS = [
    "Est-ce contagieux ?",
    "Quels soins d'urgence ?",
    "Quand consulter un médecin ?",
    "Comment éviter la propagation ?"
]

@app.post("/chat/", response_model=schemas.ChatResponse)
async def chat_endpoint(request: schemas.ChatRequest):
    """
    Endpoint chatbot médical : dialogue contextuel avec Gemma/Gemini.
    Reçoit un historique de messages et retourne une réponse médicale encadrée.
    """
    try:
        # Construction du prompt à partir de l'historique
        if not request.messages:
            return schemas.ChatResponse(reply="Bonjour, comment puis-je vous aider concernant votre santé cutanée ?")

        convo_lines = []
        for m in request.messages[-10:]:  # garder les 10 derniers messages
            role = "Utilisateur" if m.role.lower() == "user" else "Assistant"
            convo_lines.append(f"{role} : {m.content}")
        convo_text = "\n".join(convo_lines)

        # ===== RAG : récupérer un contexte lié à la dernière question =====
        last_user_text = ""
        for m in reversed(request.messages):
            if m.role.lower() == "user":
                last_user_text = m.content
                break

        rag_context = build_context(last_user_text, top_k=3)
        context_block = ""
        if rag_context:
            context_block = f"""
CONTEXTE MÉDICAL DE RÉFÉRENCE (généralement fiable, à reformuler avec tes propres mots) :
{rag_context}

"""

        prompt = f"""
{context_block}Historique de la conversation :
{convo_text}

Réponds de manière utile et éducative à la dernière question de l'utilisateur,
en t'appuyant sur le contexte fourni si pertinent.
"""

        reply = ai_provider.generate(
            system_prompt=CHAT_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=400
        )

        if not reply or len(reply) < 20:
            reply = ("Je ne peux pas vous répondre précisément pour le moment. "
                     "Si vous avez des symptômes inquiétants, consultez rapidement un professionnel de santé.")

        reply_clean = clean_markdown_for_mobile(reply)
        reply_clean += "\n\n⚠️ SkinDetect Care n'est pas un médecin. Consultez un professionnel de santé pour un diagnostic."

        return schemas.ChatResponse(reply=reply_clean)

    except Exception as e:
        print("⚠️ Chat indisponible :", e)
        return schemas.ChatResponse(
            reply="L'assistant est momentanément indisponible. Pour toute urgence, contactez directement un professionnel de santé."
        )


# ===== ENDPOINTS AUTHENTIFICATION =====
@app.post("/users/", response_model=AuthResponse)
async def create_user(
    user: schemas.UserCreate, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Crée un nouvel utilisateur et envoie un email de bienvenue"""
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = crud.create_user(db, user)
    
    # Envoi de l'email de bienvenue en arrière-plan
    background_tasks.add_task(send_welcome_email, new_user.email, new_user.username)
    
    return {
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email
        },
        "token": get_token(new_user)
    }

@app.post("/login/")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Authentifie un utilisateur"""
    user = crud.get_user_by_username(db, data.username)
    
    if not user or not auth.verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        },
        "token": get_token(user)
    }

# ===== ENDPOINT PRÉDICTION =====
@app.post("/predict/")
async def predict(
    user_id: int = Form(...),
    prediction: str = Form(...),
    confidence: float = Form(...),
    file: UploadFile = File(...),
    send_email: bool = Form(True),  # Activé par défaut
    case_file_id: int = Form(None),  # Suivi d'évolution : dossier optionnel
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Endpoint de prédiction avec conseils statiques et envoi d'email optionnel.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Générer les conseils statiques
        # final_advice = get_static_advice(prediction)
        final_advice = get_ai_advice_with_fallback(prediction, confidence)

        
        # Sauvegarde dans la base
        prediction_data = schemas.PredictionCreate(
            filename=file.filename,
            prediction=prediction,
            confidence=confidence,
            advice=final_advice,
            image=contents,
            case_file_id=case_file_id
        )
        crud.create_prediction(db, prediction_data, user_id)

        # Mise à jour automatique du dossier de maladie (suivi d'évolution)
        if case_file_id:
            case = crud.get_case_file(db, case_file_id)
            if case:
                if not case.condition:
                    case.condition = prediction
                # Statut estimé selon la confiance/gravité
                if prediction.lower() in ["measles", "monkeypox"]:
                    case.status = "worsening"
                elif prediction.lower() == "healthy":
                    if case.status != "worsening":
                        case.status = "improvement"
                db.commit()
        
        # Envoi d'email si activé
        if send_email and background_tasks:
            user = crud.get_user_by_id(db, user_id)
            if user and user.email:
                background_tasks.add_task(
                    send_prediction_email,
                    user.email,
                    user.username,
                    prediction,
                    confidence,
                    final_advice
                )

        return JSONResponse({
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence,
            "advice": final_advice,
            "email_sent": send_email
        })
        
    except Exception as e:
        print(f"❌ Erreur générale : {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ===== ENDPOINT TEST EMAIL =====
@app.post("/send-test-email/")
async def send_test_email(
    email: EmailStr,
    background_tasks: BackgroundTasks
):
    """Endpoint pour tester l'envoi d'email"""
    background_tasks.add_task(
        send_welcome_email, 
        email, 
        "Utilisateur Test"
    )
    return {
        "message": f"Email de test envoyé à {email}",
        "status": "success"
    }

# ===== ENDPOINTS RÉINITIALISATION MOT DE PASSE =====
@app.post("/auth/forgot-password/")
async def forgot_password(
    request: schemas.ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Demande un code OTP pour la réinitialisation de mot de passe"""
    user = crud.get_user_by_email(db, request.email)
    if not user:
        # Ne pas révéler si l'email existe (sécurité)
        return {"message": "Si cet email existe, un code de réinitialisation a été envoyé.", "status": "sent"}

    otp_code = crud.create_reset_otp(db, user.id)
    background_tasks.add_task(send_otp_email, user.email, user.username, otp_code)

    return {"message": "Un code de réinitialisation a été envoyé par email.", "status": "sent"}

@app.post("/auth/reset-password/")
async def reset_password(
    request: schemas.ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """Vérifie l'OTP et réinitialise le mot de passe"""
    user = crud.get_user_by_email(db, request.email)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    otp = crud.get_valid_otp(db, user.id, request.otp_code)
    if not otp:
        raise HTTPException(status_code=400, detail="Code OTP invalide, expiré ou déjà utilisé")

    # Réinitialiser et invalider l'OTP
    crud.reset_user_password(db, user.id, request.new_password)
    crud.invalidate_otp(db, otp)

    return {"message": "Mot de passe réinitialisé avec succès.", "status": "success"}

# ===== ENDPOINT STATISTIQUES =====
@app.get("/stats/summary/", response_model=schemas.StatsOverview)
def stats_summary(db: Session = Depends(get_db)):
    """Retourne un résumé des statistiques globales de la plateforme"""
    return crud.get_global_stats(db)

# ===== ENDPOINTS HISTORIQUE =====
@app.get("/history/{user_id}", response_model=List[schemas.PredictionOut])
def history(user_id: int, db: Session = Depends(get_db)):
    """Récupère l'historique des prédictions d'un utilisateur"""
    return crud.get_predictions(db, user_id)

@app.get("/predictions/unsynced/{user_id}", response_model=List[schemas.PredictionOut])
def unsynced_predictions(user_id: int, db: Session = Depends(get_db)):
    """Récupère les prédictions non synchronisées"""
    return db.query(db_models.Prediction).filter(
        db_models.Prediction.user_id == user_id
    ).all()

# ============================================================
# ===== SUIVI D'ÉVOLUTION : DOSSIERS DE MALADIE (CASE FILES) =
# ============================================================
@app.post("/case-files/create/", response_model=schemas.CaseFileOut)
def create_case_file_v2(data: schemas.CaseFileCreate, user_id: int = Form(0), db: Session = Depends(get_db)):
    """Crée un dossier de maladie (user_id passé en Form pour compatibilité Flutter)."""
    return crud.create_case_file(db, user_id, data)

@app.post("/case-files/json/", response_model=schemas.CaseFileOut)
def create_case_file_json(data: dict, db: Session = Depends(get_db)):
    """Crée un dossier de maladie en JSON : {user_id, title, condition, status}."""
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id requis")
    cf = schemas.CaseFileCreate(
        title=data.get("title", "Lésion cutanée"),
        condition=data.get("condition"),
        status=data.get("status", "stable"),
    )
    return crud.create_case_file(db, int(user_id), cf)

@app.get("/case-files/{user_id}", response_model=List[schemas.CaseFileOut])
def get_case_files(user_id: int, db: Session = Depends(get_db)):
    """Liste les dossiers de maladie d'un utilisateur."""
    return crud.get_case_files(db, user_id)

@app.get("/case-files/detail/{case_file_id}", response_model=schemas.CaseFileOut)
def get_case_file_detail(case_file_id: int, db: Session = Depends(get_db)):
    """Détail d'un dossier avec ses prédictions."""
    cf = crud.get_case_file(db, case_file_id)
    if not cf:
        raise HTTPException(status_code=404, detail="Dossier introuvable")
    return cf

@app.patch("/case-files/{case_file_id}", response_model=schemas.CaseFileOut)
def update_case_file(case_file_id: int, data: schemas.CaseFileUpdate, db: Session = Depends(get_db)):
    """Met à jour le titre/statut d'un dossier de maladie."""
    cf = crud.update_case_file(db, case_file_id, data)
    if not cf:
        raise HTTPException(status_code=404, detail="Dossier introuvable")
    return cf

@app.delete("/case-files/{case_file_id}")
def delete_case_file(case_file_id: int, db: Session = Depends(get_db)):
    """Supprime un dossier de maladie (et ses prédictions associées)."""
    crud.delete_case_file(db, case_file_id)
    return {"status": "deleted"}

@app.patch("/predictions/{prediction_id}/notes", response_model=schemas.PredictionOut)
def update_prediction_notes(prediction_id: int, data: schemas.PredictionNotesUpdate, db: Session = Depends(get_db)):
    """Ajoute/mode des notes de suivi sur une prédiction."""
    pred = crud.update_prediction_notes(db, prediction_id, data.notes)
    if not pred:
        raise HTTPException(status_code=404, detail="Prédiction introuvable")
    return pred

# ============================================================
# ===== QUESTIONNAIRE ANANMESTIQUE + IA ======================
# ============================================================
@app.post("/anamnesis/", response_model=schemas.AnamnesisResponse)
def anamnesis_endpoint(req: schemas.AnamnesisRequest, db: Session = Depends(get_db)):
    """
    Combine la prédiction visuelle avec un questionnaire (démangeaisons, douleur,
    ancienneté, antécédents) pour affiner les conseils via le provider IA.
    """
    try:
        q = req.questions or {}
        itchy = q.get("itchy")
        pain = q.get("pain")
        duration = q.get("duration_days")
        history = q.get("history")
        location = q.get("location")

        q_lines = []
        q_lines.append(f"- Démangeaisons : {'oui' if itchy else 'non' if itchy is not None else 'non précisé'}")
        q_lines.append(f"- Douleur : {'oui' if pain else 'non' if pain is not None else 'non précisé'}")
        q_lines.append(f"- Ancienneté : {duration} jour(s) " if duration is not None else "- Ancienneté : non précisée")
        if location:
            q_lines.append(f"- Localisation : {location}")
        if history:
            q_lines.append(f"- Antécédents / contexte : {history}")
        questionnaire_text = "\n".join(q_lines)

        rag_context = build_context((req.prediction or "") + " " + (str(history) or ""), top_k=3)
        context_block = ""
        if rag_context:
            context_block = f"\nCONTEXTE MÉDICAL DE RÉFÉRENCE (à reformuler) :\n{rag_context}\n"

        system = (
            "Tu es un assistant médical éducatif en dermatologie. Tu n'es pas un médecin et tu "
            "n'établis aucun diagnostic. Réponds en français, de façon simple et rassurante. "
            "Croise une prédiction d'analyse d'image avec les réponses du patient pour donner des "
            "conseils personnalisés. Ne prescris jamais de médicaments."
        )
        prompt = (
            f"{context_block}"
            f"Prédiction du modèle d'image : {req.prediction} (confiance {req.confidence*100:.1f}%)\n\n"
            f"Réponses au questionnaire anamnestique :\n{questionnaire_text}\n\n"
            "Avec ces informations, donne des conseils personnalisés expliquant :\n"
            "1. Ce que cela évoque\n"
            "2. Les gestes simples à faire en tenant compte des réponses\n"
            "3. Quand consulter un médecin\n"
            "4. Un message rassurant"
        )

        reply = ai_provider.generate(
            system_prompt=system,
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=400,
        )

        if not reply or len(reply) < 30:
            return schemas.AnamnesisResponse(refined_advice=get_static_advice(req.prediction))

        return schemas.AnamnesisResponse(refined_advice=clean_markdown_for_mobile(reply))
    except Exception as e:
        print("⚠️ Anamnèse indisponible :", e)
        return schemas.AnamnesisResponse(refined_advice=get_static_advice(req.prediction))

# ============================================================
# ===== QUALITÉ D'IMAGE (côté serveur) ========================
# ============================================================
@app.post("/image-quality/", response_model=schemas.ImageQualityResponse)
async def image_quality_endpoint(file: UploadFile = File(...)):
    """Analyse la qualité d'une photo envoyée (luminosité, netteté)."""
    contents = await file.read()
    result = imgq.analyze_quality(contents)
    return schemas.ImageQualityResponse(**result)

# ============================================================
# ===== RAPPORT PDF CÔTÉ SERVEUR + PARTAGE EMAIL =============
# ============================================================
@app.post("/report/pdf/", response_model=schemas.PdfResponse)
async def generate_report_pdf(req: schemas.PdfRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Génère un PDF (rapport d'une prédiction ou d'un dossier complet) côté serveur."""
    user = crud.get_user_by_id(db, req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    patient_name = user.username
    pdf_bytes = None
    prediction_fr = ""

    if req.case_file_id:
        cf = crud.get_case_file(db, req.case_file_id)
        if not cf:
            raise HTTPException(status_code=404, detail="Dossier introuvable")
        preds = cf.predictions
        pdf_bytes = report_pdf.build_casefile_pdf(patient_name, cf, preds)
    elif req.prediction_id:
        p = crud.get_prediction_by_id(db, req.prediction_id)
        if not p:
            raise HTTPException(status_code=404, detail="Prédiction introuvable")
        prediction_fr = STATIC_ADVICE.get(p.prediction.lower(), STATIC_ADVICE["default"])["description"]
        pdf_bytes = report_pdf.build_prediction_pdf(
            patient_name, p.prediction, prediction_fr,
            p.confidence, p.advice, p.notes, p.timestamp
        )
    else:
        # Par défaut : dernier examen
        preds = crud.get_predictions(db, req.user_id)
        if not preds:
            raise HTTPException(status_code=404, detail="Aucune prédiction disponible")
        p = preds[0]
        prediction_fr = STATIC_ADVICE.get(p.prediction.lower(), STATIC_ADVICE["default"])["description"]
        pdf_bytes = report_pdf.build_prediction_pdf(
            patient_name, p.prediction, prediction_fr,
            p.confidence, p.advice, p.notes, p.timestamp
        )

    response = schemas.PdfResponse(pdf_base64=base64.b64encode(pdf_bytes).decode())

    if req.send_email_to:
        background_tasks.add_task(send_report_email, req.send_email_to, patient_name, pdf_bytes)
        response.emailed_to = str(req.send_email_to)

    return response

# ============================================================
# ===== RAPPELS PERSONNALISÉS =================================
# ============================================================
@app.post("/reminders/", response_model=schemas.ReminderOut)
def create_reminder(data: schemas.ReminderCreate, db: Session = Depends(get_db)):
    """Crée un rappel programmé."""
    return crud.create_reminder(db, data)

@app.get("/reminders/{user_id}", response_model=List[schemas.ReminderOut])
def get_reminders(user_id: int, db: Session = Depends(get_db)):
    """Liste les rappels d'un utilisateur."""
    return crud.get_reminders(db, user_id)

@app.patch("/reminders/{reminder_id}", response_model=schemas.ReminderOut)
def update_reminder(reminder_id: int, data: schemas.ReminderUpdate, db: Session = Depends(get_db)):
    r = crud.update_reminder(db, reminder_id, data)
    if not r:
        raise HTTPException(status_code=404, detail="Rappel introuvable")
    return r

@app.delete("/reminders/{reminder_id}")
def delete_reminder(reminder_id: int, db: Session = Depends(get_db)):
    crud.delete_reminder(db, reminder_id)
    return {"status": "deleted"}

# ============================================================
# ===== RGPD : GESTION DU CONSENTEMENT ET DES DONNÉES ========
# ============================================================
@app.post("/account/{user_id}/consent")
def update_consent(user_id: int, data: schemas.ConsentUpdate, db: Session = Depends(get_db)):
    """Définit le consentement au stockage des données de santé."""
    user = crud.update_consent(db, user_id, data.consent_health_data)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {"consent_health_data": user.consent_health_data, "consent_at": user.consent_at}

@app.get("/account/{user_id}/export", response_model=schemas.ExportOut)
def export_account(user_id: int, db: Session = Depends(get_db)):
    """Export complet des données de l'utilisateur (droit à la portabilité)."""
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    preds = crud.get_predictions(db, user_id)
    cfs = crud.get_case_files(db, user_id)
    return schemas.ExportOut(
        user_id=user.id,
        username=user.username,
        email=user.email,
        consent_health_data=user.consent_health_data or False,
        anonymized=user.anonymized or False,
        predictions_count=len(preds),
        case_files_count=len(cfs),
        predictions=preds,
        case_files=cfs,
    )

@app.post("/account/{user_id}/anonymize")
def anonymize_account(user_id: int, db: Session = Depends(get_db)):
    """Anonymise le compte : remplace les données identifiantes mais conserve les statistiques."""
    user = crud.anonymize_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {"status": "anonymized", "username": user.username, "email": user.email}

@app.delete("/account/{user_id}")
def delete_account(user_id: int, db: Session = Depends(get_db)):
    """Supprime définitivement le compte et toutes ses données (droit à l'oubli)."""
    crud.delete_user(db, user_id)
    return {"status": "deleted"}

# ===== ENDPOINTS UTILISATEURS =====
@app.get("/users/{user_id}", response_model=schemas.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son ID"""
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/username/{username}", response_model=schemas.UserOut)
def get_user_by_username(username: str, db: Session = Depends(get_db)):
    """Récupère un utilisateur par son nom d'utilisateur"""
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/", response_model=List[schemas.UserOut])
def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Récupère tous les utilisateurs (avec pagination)"""
    return crud.get_users(db, skip=skip, limit=limit)

# ===== ENDPOINT MALADIES SUPPORTÉES =====
@app.get("/supported-conditions/")
def get_supported_conditions():
    """Retourne la liste des conditions avec conseils statiques disponibles"""
    conditions = []
    for key, value in STATIC_ADVICE.items():
        if key != "default":
            conditions.append({
                "name": key,
                "description": value["description"]
            })
    return {"conditions": conditions, "total": len(conditions)}

# ===== HEALTH CHECK =====
@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "SkinDetect API - Version avec emails",
        "version": "1.0.0",
        "features": ["predictions", "email_notifications", "history"],
        "ml_inference": "Client-side (Flutter)"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "memory_usage": "< 200MB",
        "email_configured": bool(os.getenv("MAIL_PASSWORD"))
    }

# ===== KEEP-ALIVE POUR RENDER =====
import threading
import requests

def keep_alive():
    """Ping l'API toutes les 2 minutes pour éviter la mise en veille"""
    while True:
        try:
            url = "https://detectionbackend-hln7.onrender.com"
            if url:
                print(f"[Keep-alive] Ping vers {url}")
                requests.get(url + "/health", timeout=2)
        except Exception as e:
            print(f"[Keep-alive] Erreur: {e}")
        
        time.sleep(120)  # 2 minutes

# Lancer le keep-alive dans un thread séparé
threading.Thread(target=keep_alive, daemon=True).start()

# ===== SCHEDULER RAPPELS =====
import asyncio as _asyncio

def _run_reminder_loop(db_session_factory):
    """Vérifie périodiquement les rappels arrivés à échéance et envoie les emails."""
    while True:
        try:
            db = db_session_factory()
            now = datetime.datetime.utcnow()
            due = crud.get_due_reminders(db, now)
            for r in due:
                user = crud.get_user_by_id(db, r.user_id)
                if user and user.email:
                    _asyncio.run(send_reminder_email(user.email, r.title, r.message))
                # Rééchelonner ou marquer comme envoyé
                if r.frequency == "daily":
                    r.remind_at = now + datetime.timedelta(days=1)
                elif r.frequency == "weekly":
                    r.remind_at = now + datetime.timedelta(weeks=1)
                else:  # once
                    r.status = "sent"
                db.commit()
            db.close()
        except Exception as e:
            print(f"[Rappels] Erreur: {e}")
        time.sleep(60)  # vérifie toutes les minutes

threading.Thread(target=_run_reminder_loop, args=(database.SessionLocal,), daemon=True).start()

# ===== DÉMARRAGE =====
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)