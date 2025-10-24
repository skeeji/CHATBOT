1. Base image Python officielle

FROM python:3.11-slim

2. D�finir le r�pertoire de travail

WORKDIR /app

3. Copier et installer les d�pendances D'ABORD

Cela permet � Docker de mettre en cache cette �tape

COPY requirements.txt .

Installer torch depuis l'index CPU pour une image beaucoup plus l�g�re

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

4. T�l�charger le mod�le PENDANT le build Docker

Cela place le mod�le dans /app/models/all-MiniLM-L6-v2

RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); model.save('/app/models/all-MiniLM-L6-v2')"

5. Copier les fichiers de l'application (data, static, code)

(Le .dockerignore emp�chera la copie des fichiers inutiles)

COPY . .

6. Port Cloud Run

ENV PORT 8080

7. Lancer Gunicorn

--- CORRECTION CRITIQUE ---

Votre fichier s'appelle "main.py", donc nous utilisons "main:app"

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 300 main:app