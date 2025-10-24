# --- �TAPE 1: Base image Python ---
FROM python:3.11-slim

# --- �TAPE 2: D�finir le r�pertoire de travail ---
WORKDIR /app

# --- �TAPE 3: Installer les d�pendances ---
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- �TAPE 4: T�l�charger le NOUVEAU mod�le dans l'image ---
# ATTENTION: Cette �tape sera BEAUCOUP plus longue car le mod�le est plus gros
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); model.save('/app/models/paraphrase-multilingual-mpnet-base-v2')"

# --- �TAPE 5: Copier TOUT le code de l'application ---
COPY . .

# --- �TAPE 6: D�finir le port ---
ENV PORT 8080

# --- �TAPE 7: Lancer l'application ---
# On augmente le timeout � 600s car le nouveau mod�le est plus lent � charger
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 600 main:app