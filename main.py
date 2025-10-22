import os
import logging
import pickle
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# Note: Les imports pour la recherche par image (ex: PIL, scipy) sont laissés en commentaire.

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
TEXT_EMBEDDINGS_PATH = 'data/text_embeddings_free.pkl'

# --- FONCTION DE CHARGEMENT DE BASE DE DONNÉES ---

def load_database():
    """Chargement de toutes les bases de données (Image et Texte) au démarrage de l'API."""
    # Déclarez toutes les variables globales que vous modifiez
    global features_array, image_ids, metadata_list, text_features_array, text_metadata_list, TEXT_EMBEDDING_MODEL
    
    logger.info("🚀 DÉBUT CHARGEMENT BASE DE DONNÉES")
    
    # 1. CHARGEMENT BASE D'IMAGES (Si vous avez vos fichiers)
    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH}...")
            # --- Votre logique de chargement d'embeddings d'images ici ---
            # Exemple:
            # with open(IMAGE_EMBEDDINGS_PATH, 'rb') as f:
            #     data = pickle.load(f)
            #     features_array = np.array(data['features'])
            #     image_ids = data['image_ids']
            #     metadata_list = data['metadata']
            logger.info("✅ Embeddings Images chargés. (Vérifiez les logs pour la forme si implémenté)")
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings images: {str(e)}")

    # 2. CHARGEMENT BASE TEXTUELLE GRATUITE (NÉCESSAIRE pour /api/search_text)
    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH}...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte Gratuits chargés: {text_features_array.shape}")
            
            # Charger le modèle pour les requêtes utilisateur
            logger.info(f"🧠 Chargement du modèle {MODEL_NAME_FREE} pour les requêtes utilisateur...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu' 
            TEXT_EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME_FREE, device=device)
            logger.info(f"✅ Modèle de requête Texte chargé sur {device}.")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings texte ou modèle: {str(e)}")
            text_features_array = None
            text_metadata_list = None
            TEXT_EMBEDDING_MODEL = None
    else:
        logger.warning(f"⚠️ FICHIER TEXTUEL GRATUIT MANQUANT: {TEXT_EMBEDDINGS_PATH}. Recherche texte désactivée.")
        
    return True

# --- FONCTIONS DE RECHERCHE TEXTUELLE ---

def extract_text_features(text_query):
    # ... (Le corps de cette fonction reste inchangé par rapport à votre version)
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
    # ... (Le corps de cette fonction reste inchangé par rapport à votre version)
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

# Route de Santé (Health Check) - Pour vérifier que l'API est vivante
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
    # Attention: 'static/images' doit être le dossier où se trouvent vos images
    return send_from_directory('static/images', filename)

# TODO: INSEREZ ICI VOTRE ROUTE DE RECHERCHE PAR IMAGE EXISTANTE (/api/search)
# @app.route('/api/search', methods=['POST', 'OPTIONS'])
# def api_search_image():
#     # ... Votre code existant pour la recherche par image ...
#     pass


# --- LANCEMENT DE L'APPLICATION (Pour le test local uniquement) ---
if __name__ == '__main__':
    # Ne devrait pas être utilisé par Cloud Run, mais pour le test local
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
