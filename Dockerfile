# --- �TAPE 1: Base image Python ---
FROM python:3.11-slim

# --- �TAPE 2: D�finir le r�pertoire de travail ---
WORKDIR /app

# --- �TAPE 3: Installer les d�pendances ---
# Copier requirements.txt d'abord pour le cache Docker
COPY requirements.txt .

# Installer torch-cpu pour une image plus l�g�re
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Installer le reste
RUN pip install --no-cache-dir -r requirements.txt

# --- �TAPE 4: T�l�charger le mod�le dans l'image ---
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

# --- �TAPE 5: Copier TOUT le code de l'application ---
# (Le .dockerignore va filtrer les fichiers inutiles)
COPY . .

# --- �TAPE 6: D�finir le port ---
ENV PORT 8080

# --- �TAPE 7: Lancer l'application ---
# Commande correcte : "main:app" car votre fichier est main.py
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 main:app