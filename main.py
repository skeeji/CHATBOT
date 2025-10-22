import os
import logging
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# ATTENTION : Si vous utilisez Flask-CORS, assurez-vous qu'il est bien dans requirements.txt

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VARIABLES GLOBALES ---
# Variables pour la recherche par image (Vecteurs pour la recherche visuelle)
features_array = None 
image_ids = None      
metadata_list = None  
IMAGE_EMBEDDINGS_PATH = 'data/embeddings.pkl' 

# Variables pour la recherche textuelle GRATUITE (Vecteurs pour la recherche sémantique)
text_features_array = None  
text_metadata_list = None   
TEXT_EMBEDDING_MODEL = None 
MODEL_NAME_FREE = "all-MiniLM-L6-v2"
# Le chemin vers le PKL maintenant sur GitHub
TEXT_EMBEDDINGS_PATH = 'data/text_embeddings_free.pkl' 

# --- FONCTION DE CHARGEMENT DE BASE DE DONNÉES ---

def load_database():
    """
    Charge les bases de données (images et texte) depuis les fichiers locaux.
    Ceci est appelé une fois au démarrage par Gunicorn.
    """
    # Déclarez toutes les variables globales que vous modifiez
    global features_array, image_ids, metadata_list, text_features_array, text_metadata_list, TEXT_EMBEDDING_MODEL
    
    logger.info("🚀 DÉBUT CHARGEMENT BASE DE DONNÉES DEPUIS FICHIERS LOCAUX")
    
    # 1. CHARGEMENT BASE TEXTUELLE CRITIQUE
    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH}...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte Gratuits chargés: {text_features_array.shape}")
            
            # Charger le modèle pour les requêtes utilisateur (étape gourmande en ressources)
            logger.info(f"🧠 Chargement du modèle {MODEL_NAME_FREE} pour les requêtes utilisateur...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu' 
            TEXT_EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME_FREE, device=device)
            logger.info(f"✅ Modèle de requête Texte chargé sur {device}.")
            
        except Exception as e:
            logger.error(f"❌ Erreur FATALE lors du chargement ou de l'initialisation du modèle: {str(e)}")
            text_features_array = None
            text_metadata_list = None
            TEXT_EMBEDDING_MODEL = None
    else:
        logger.error(f"⚠️ FICHIER TEXTUEL CRITIQUE MANQUANT: {TEXT_EMBEDDINGS_PATH}. L'API de recherche texte est DÉSACTIVÉE.")
        
    # 2. CHARGEMENT BASE D'IMAGES (Si vous l'utilisez, sinon les variables resteront None)
    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH}...")
        # (Logique de chargement d'images ici si nécessaire)
    else:
        logger.warning(f"⚠️ FICHIER IMAGE MANQUANT: {IMAGE_EMBEDDINGS_PATH}")
        
    logger.info("🏁 Fin du chargement de la base de données.")
    return True

# --- FONCTIONS DE RECHERCHE TEXTUELLE (INCHANGÉES) ---

def extract_text_features(text_query):
    # ... (Le corps de cette fonction reste inchangé)
    global TEXT_EMBEDDING_MODEL
    
    if TEXT_EMBEDDING_MODEL is None:
        logger.error("❌ Modèle Sentence Transformer non chargé. Impossible de vectoriser la requête.")
        return None
        
    try:
        logger.info(f"🧠 Extraction features texte local pour: '{text_query[:30]}...'")
        
        features = TEXT_EMBEDDING_MODEL.encode(
            text_query, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        return features.flatten()
            
    except Exception as e:
        logger.error(f"💥 Extraction Texte Error: {str(e)}")
        return None

def search_logic_text(query, top_k=10):
    # ... (Le corps de cette fonction reste inchangé)
    global text_features_array, text_metadata_list
    
    if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
        return {'success': False, 'error': 'Service de recherche texte non prêt (base de données ou modèle non chargé)'}, 503
        
    # 1. Extraire le vecteur de la requête utilisateur
    query_features = extract_text_features(query)
    if query_features is None:
        return {'success': False, 'error': 'Impossible de vectoriser la requête'}, 500
    
    # 2. Calculer similarités
    logger.info("🧮 Calcul des similarités textuelles...")
        
    if text_features_array.shape[1] != query_features.shape[0]:
        logger.error(f"Incompatibilité de dimension : DB {text_features_array.shape[1]} vs Query {query_features.shape[0]}")
        return {'success': False, 'error': 'Incompatibilité de dimension de vecteur'}, 500
        
    similarity_scores = np.dot(text_features_array, query_features)
    
    # 3. Trier les résultats
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    top_results = []
    for rank in range(min(top_k, len(sorted_indices))):
        i = sorted_indices[rank]
        score = similarity_scores[i]
        
        if i >= len(text_metadata_list):
            continue
            
        metadata = text_metadata_list[i]
        
        top_results.append({
            'index': int(i),
            'similarity': float(score),
            'nom': metadata.get('nom', 'N/A'),
            'artiste': metadata.get('artiste', 'N/A'),
            'annee': metadata.get('annee', 'N/A'),
            'image_url': f"/images/{metadata.get('image_id', '')}", 
            'lien_site': metadata.get('lien_site', 'N/A')
        })

    top_scores = [f'{s["similarity"]:.3f}' for s in top_results[:3]]
    logger.info("✅ Top 3 similarités texte: {}".format(top_scores))
    
    return {
        'success': True,
        'query_processed': True,
        'total_database_size': len(text_features_array) if text_features_array is not None else 0,
        'results': top_results
    }, 200


# --- INITIALISATION DE L'APPLICATION FLASK ---

app = Flask(__name__, static_folder='static')
CORS(app)

# ⚡️ CHARGEMENT IMMÉDIAT DES MODÈLES
# Gunicorn appellera cette fonction une fois au démarrage de chaque worker.
load_database()

# --- ROUTES D'API ---

# Route de Base (DEBUG)
@app.route('/', methods=['GET'])
def home():
    """Route de base simple pour tester si Flask répond."""
    return jsonify({
        'status': 'Bienvenue',
        'message': 'API en ligne. Testez /health ou /api/search_text',
        'db_ready': text_features_array is not None
    })

# Route de Santé (Health Check)
@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie si le service est prêt (modèles chargés)."""
    status = 'ok' if text_features_array is not None and TEXT_EMBEDDING_MODEL is not None else 'loading'
    
    if status == 'ok':
        message = 'Service de recherche sémantique prêt.'
    else:
        message = 'Chargement en cours ou échec du chargement de la base de données/modèle.'
        logger.warning("⚠️ Health Check: Modèle non prêt.")
        
    return jsonify({
        'status': status,
        'message': message,
        'db_ready': text_features_array is not None
    }), 200 if status == 'ok' else 503


# Route pour la recherche sémantique textuelle (via le chatbot)
@app.route('/api/search_text', methods=['POST', 'OPTIONS'])
def api_search_text():
    """ROUTE PRINCIPALE POUR LA RECHERCHE SÉMANTIQUE TEXTUELLE"""
    try:
        logger.info("🔍 API SEARCH TEXT ENDPOINT")
        
        if request.method == 'OPTIONS':
            # Gère le preflight CORS
            response = jsonify({'status': 'OK'})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response
            
        data = request.json
        query = data.get('query')
        top_k = int(data.get('top_k', 10))
        
        if not query:
            return jsonify({'success': False, 'error': 'Requête de recherche manquante'}), 400
        
        result, status_code = search_logic_text(query, top_k)
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"💥 API Search Text Error: {type(e).__name__}: {str(e)}")
        return jsonify({'success': False, 'error': 'Erreur interne du serveur'}), 500

# Route pour servir les images (essentielle pour le frontend)
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sert les images stockées dans static/images/"""
    # ATTENTION: 'static/images' doit être le dossier où se trouvent vos images
    return send_from_directory('static/images', filename)


# --- LANCEMENT DE L'APPLICATION (Pour le test local uniquement) ---
if __name__ == '__main__':
    # Ne devrait pas être utilisé par Cloud Run, mais pour le test local
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
