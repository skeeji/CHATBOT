# --- ÉTAPE 1: Base image Python ---
FROM python:3.11-slim

# --- ÉTAPE 2: Définir le répertoire de travail ---
WORKDIR /app

# --- ÉTAPE 3: Installer les dépendances ---
# Copier requirements.txt d'abord pour le cache Docker
COPY requirements.txt .

# Installer torch-cpu pour une image plus légère
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Installer le reste
RUN pip install --no-cache-dir -r requirements.txt

# --- ÉTAPE 4: Télécharger le modèle dans l'image ---
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

# --- ÉTAPE 5: Copier TOUT le code de l'application ---
# (Le .dockerignore va filtrer les fichiers inutiles)
COPY . .

# --- ÉTAPE 6: Définir le port ---
ENV PORT 8080

# --- ÉTAPE 7: Lancer l'application ---
# Commande correcte : "main:app" car votre fichier est main.py
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 main:app