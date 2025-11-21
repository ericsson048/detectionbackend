Project Analysis: Skin Disease Detection Backend
Overview
This project is a backend API for detecting skin diseases (specifically Mpox, Chickenpox, Measles, etc.) from images. It uses a deep learning model (ResNet18) and provides advice using Google's Gemini AI.

Tech Stack
Framework: FastAPI
Language: Python 3.x
Database: PostgreSQL (via SQLAlchemy ORM)
ML/AI:
PyTorch (ResNet18 model)
Google Gemini (Generative AI for advice)
Authentication: JWT (JSON Web Tokens)
Deployment: Uvicorn
Key Components
1. API (
main.py
)
Endpoints:
POST /users/: Register a new user.
POST /login: Authenticate and get a JWT.
POST /predict/: Upload an image for disease detection. Returns prediction, confidence, and AI-generated advice.
GET /history/{user_id}: Retrieve prediction history for a user.
GET /users/{user_id}: Get user details.
2. Database (
database.py
, 
db_models.py
, 
crud.py
)
Models:
User
: Stores credentials and profile info.
Prediction
: Stores image metadata, prediction result, confidence score, and advice.
Configuration: Connects to a PostgreSQL database (default: mpoxdb on localhost).
3. Machine Learning (
main.py
, 
detect.py
)
Inference: 
main.py
 loads a pre-trained ResNet18 model (
best_mpox_model.pth
) to classify images into 6 categories: Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox.
Training: 
detect.py
 is a standalone script used to train the model using a dataset from Roboflow.
4. Authentication (
auth.py
)
Uses bcrypt for password hashing.
Generates JWT tokens with a 7-day expiration.
5. External Integrations
Google Gemini: Used in 
predict
 endpoint to generate context-aware medical advice based on the prediction.
Roboflow: Used in 
detect.py
 for dataset management.
Configuration
Environment Variables:
DATABASE_URL: Connection string for PostgreSQL.
GEMINI_API_KEY: API key for Google Gemini.
PORT: Port for the application (default: 10000).
Observations
The project is well-structured with separation of concerns (schemas, models, crud, auth).
detect.py
 is for training and not used at runtime.
The API is ready for deployment with CORS configured.