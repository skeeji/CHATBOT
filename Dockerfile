# --- ÉTAPE 1: Base image Python ---
FROM python:3.11-slim

# --- ÉTAPE 2: Définir le répertoire de travail ---
WORKDIR /app

# --- ÉTAPE 3: Installer les dépendances ---
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- ÉTAPE 4: Télécharger le NOUVEAU modèle dans l'image ---
# ATTENTION: Cette étape sera BEAUCOUP plus longue car le modèle est plus gros
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); model.save('/app/models/paraphrase-multilingual-mpnet-base-v2')"

# --- ÉTAPE 5: Copier TOUT le code de l'application ---
COPY . .

# --- ÉTAPE 6: Définir le port ---
ENV PORT 8080

# --- ÉTAPE 7: Lancer l'application ---
# On augmente le timeout à 600s car le nouveau modèle est plus lent à charger
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 600 main:app