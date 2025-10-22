#!/bin/bash

# Script de déploiement automatique pour Cloud Run
# Usage: ./deploy.sh

set -e

echo "🚀 Début du déploiement Cloud Run..."

# Variables (à adapter si nécessaire)
PROJECT_ID=\luminaires-search-2024
REGION="europe-west1"
SERVICE_NAME="chatbot-app"
REPO_NAME="chatbot-repo"

echo "📋 Configuration:"
echo "  - Projet: \"
echo "  - Région: \"
echo "  - Service: \"
echo ""

# 1. Vérifier que le dépôt Artifact Registry existe
echo "🔍 Vérification du dépôt Artifact Registry..."
if ! gcloud artifacts repositories describe \ --location=\ &>/dev/null; then
    echo "⚠️  Le dépôt n'existe pas. Création en cours..."
    gcloud artifacts repositories create \ \
        --repository-format=docker \
        --location=\ \
        --description="Docker repository for chatbot"
    echo "✅ Dépôt créé avec succès"
else
    echo "✅ Dépôt déjà existant"
fi

# 2. Configurer l'authentification Docker
echo "🔐 Configuration de l'authentification Docker..."
gcloud auth configure-docker \-docker.pkg.dev --quiet

# 3. Build et déploiement
echo "🏗️  Construction et déploiement..."
gcloud run deploy \ \
    --source . \
    --region=\ \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --port=8080

echo ""
echo "✅ Déploiement terminé avec succès!"
echo "🌐 Votre application est accessible à l'URL affichée ci-dessus"
