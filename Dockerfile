# Base image Python officielle
FROM python:3.11-slim

# D�finir le r�pertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des d�pendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application (main.py, data/, static/, etc.)
COPY . .

# Cloud Run �coute sur le port 8080 par d�faut
ENV PORT 8080

# Ex�cuter l'application avec Gunicorn, en utilisant la variable PORT
# (main:app) lance l'application 'app' dans le fichier 'main.py'
CMD ["gunicorn", "--bind", "0.0.0.0:$(PORT)", "main:app"]
