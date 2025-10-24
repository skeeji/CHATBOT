# Base image Python officielle
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle PENDANT le build Docker
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

# Copier le code de l'application
COPY . .

# Port Cloud Run
ENV PORT 8080

# Lancer Gunicorn
CMD gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 300 main:app