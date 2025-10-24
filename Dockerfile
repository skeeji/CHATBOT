1. Base image Python officielle

FROM python:3.11-slim

2. Définir le répertoire de travail

WORKDIR /app

3. Copier et installer les dépendances D'ABORD

Cela permet à Docker de mettre en cache cette étape

COPY requirements.txt .

Installer torch depuis l'index CPU pour une image beaucoup plus légère

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

4. Télécharger le modèle PENDANT le build Docker

Cela place le modèle dans /app/models/all-MiniLM-L6-v2

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

5. Copier les fichiers de l'application (data, static, code)

(Le .dockerignore empêchera la copie des fichiers inutiles)

COPY . .

6. Port Cloud Run

ENV PORT 8080

7. Lancer Gunicorn

--- CORRECTION CRITIQUE ---

Votre fichier s'appelle "main.py", donc nous utilisons "main:app"

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 main:app