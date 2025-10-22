import os
import logging
import pickle
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
# Importez les autres bibliothèques nécessaires pour votre recherche par image (ex: PIL, scipy, etc.)
# import ...

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VARIABLES GLOBALES ---
# Variables pour la recherche par image (Gardez vos variables existantes)
features_array = None # Les embeddings d'images (vecteurs)
image_ids = None      # Les noms de fichiers d'images
metadata_list = None  # Les métadonnées d'images
IMAGE_EMBEDDINGS_PATH = 'data/embeddings.pkl' # Votre chemin existant

# Variables pour la recherche textuelle GRATUITE (NOUVEAU)
text_features_array = None  # Les embeddings de texte (vecteurs)
text_metadata_list = None   # Les métadonnées du catalogue pour le texte
TEXT_EMBEDDING_MODEL = None # Le modèle Sentence-Transformer chargé
MODEL_NAME_FREE = "all-MiniLM-L6-v2"
TEXT_EMBEDDINGS_PATH = 'data/text_embeddings_free.pkl'

# --- FONCTION DE CHARGEMENT DE BASE DE DONNÉES ---

def load_database():
    """Chargement de toutes les bases de données (Image et Texte) au démarrage de l'API."""
    # Déclarez toutes les variables globales que vous modifiez
    global features_array, image_ids, metadata_list, text_features_array, text_metadata_list, TEXT_EMBEDDING_MODEL
    
    logger.info("🚀 DÉBUT CHARGEMENT BASE DE DONNÉES")
    
    # 1. CHARGEMENT BASE D'IMAGES (VOTRE CODE EXISTANT)
    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH}...")
            
            # TODO: INSEREZ ICI VOTRE LOGIQUE DE CHARGEMENT DES EMBEDDINGS D'IMAGES
            # Exemple :
            # with open(IMAGE_EMBEDDINGS_PATH, 'rb') as f:
            #     data = pickle.load(f)
            #     features_array = np.array(data['features'])
            #     image_ids = data['image_ids']
            #     metadata_list = data['metadata']
            
            # Après le chargement (Décommentez quand votre code est là)
            # logger.info(f"✅ Embeddings Images chargés: {features_array.shape}")
            logger.info("✅ Embeddings Images chargés.")
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings images: {str(e)}")
    else:
        logger.warning(f"⚠️ FICHIER IMAGE MANQUANT: {IMAGE_EMBEDDINGS_PATH}")

    # 2. CHARGEMENT BASE TEXTUELLE GRATUITE (NOUVEAU)
    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH}...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte Gratuits chargés: {text_features_array.shape}")
            
            # Charger le modèle pour les requêtes utilisateur (optimisé pour le CPU de Vercel)
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
    """Extraction du vecteur de la requête textuelle en utilisant le modèle Sentence-Transformer local."""
    global TEXT_EMBEDDING_MODEL
    
    if TEXT_EMBEDDING_MODEL is None:
         logger.error("❌ Modèle Sentence Transformer non chargé. Impossible de vectoriser la requête.")
         return None
         
    try:
        logger.info(f"🧠 Extraction features texte local pour: '{text_query[:30]}...'")
        
        # Le modèle encode la requête et retourne le vecteur normalisé
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
    """Logique de recherche sémantique textuelle."""
    global text_features_array, text_metadata_list
    
    if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
        return {'success': False, 'error': 'Service de recherche texte non prêt (base de données ou modèle non chargé)'}, 503
        
    # 1. Extraire le vecteur de la requête utilisateur
    query_features = extract_text_features(query)
    if query_features is None:
        return {'success': False, 'error': 'Impossible de vectoriser la requête'}, 500
    
    # 2. Calculer similarités (produit scalaire entre vecteurs normalisés = Cosine Similarity)
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
            # Construction de l'URL pour le frontend via la route /images
            'image_url': f"/images/{metadata.get('image_id', '')}", 
            'lien_site': metadata.get('lien_site', 'N/A')
        })

    top_scores = [f'{s["similarity"]:.3f}' for s in top_results[:3]]
    logger.info("✅ Top 3 similarités texte: {}".format(top_scores))
    
    return {
        'success': True,
        'query_processed': True,
        'total_database_size': len(text_features_array),
        'results': top_results
    }, 200

# --- INITIALISATION DE L'APPLICATION FLASK ---
# Assurez-vous que le chemin 'static' est correct pour vos images
app = Flask(__name__, static_folder='static')
CORS(app)

# --- ROUTES D'API ---

# Nouvelle route pour la recherche sémantique textuelle (via le chatbot)
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
#     # ... Votre code existant pour la recherche par image ...
#     pass


# --- LANCEMENT DE L'APPLICATION (Pour le test local uniquement) ---
if __name__ == '__main__':
    load_database()
    app.run(debug=True, port=5000)

# Lors du déploiement sur Vercel, Vercel appelle la fonction load_database() au démarrage de l'instance.