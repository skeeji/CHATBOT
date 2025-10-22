import os
import logging
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from functools import wraps

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VARIABLES GLOBALES ---
# Variables pour la recherche par image (Gardez vos variables existantes)
features_array = None   # Les embeddings d'images (vecteurs)
image_ids = None        # Les noms de fichiers d'images
metadata_list = None    # Les métadonnées d'images
IMAGE_EMBEDDINGS_PATH = 'data/embeddings.pkl' # Votre chemin existant

# Variables pour la recherche textuelle GRATUITE
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
    
    # 1. CHARGEMENT BASE D'IMAGES
    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH} (Images)...")
            
            # --- CODE DE CHARGEMENT DES EMBEDDINGS D'IMAGES (AJOUTÉ POUR COMPLÉTER) ---
            with open(IMAGE_EMBEDDINGS_PATH, 'rb') as f:
                 data = pickle.load(f)
                 features_array = np.array(data['features'])
                 image_ids = data['image_ids']
                 metadata_list = data['metadata']
            # --------------------------------------------------------------------------
            
            logger.info(f"✅ Embeddings Images chargés: {features_array.shape}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings images: {str(e)}")
            features_array = None
    else:
        logger.warning(f"⚠️ FICHIER IMAGE MANQUANT: {IMAGE_EMBEDDINGS_PATH}. Recherche image désactivée.")

    # 2. CHARGEMENT BASE TEXTUELLE GRATUITE
    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH} (Texte)...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte Gratuits chargés: {text_features_array.shape}")
            
            # Charger le modèle pour les requêtes utilisateur
            logger.info(f"🧠 Chargement du modèle {MODEL_NAME_FREE} pour les requêtes utilisateur...")
            # Cloud Run est plus stable avec 'cpu' pour ces modèles
            device = 'cpu'
            TEXT_EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME_FREE, device=device)
            logger.info(f"✅ Modèle de requête Texte chargé sur {device}.")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement embeddings texte ou modèle: {str(e)}")
            text_features_array = None
            text_metadata_list = None
            TEXT_EMBEDDING_MODEL = None
    else:
        logger.warning(f"⚠️ FICHIER TEXTUEL GRATUIT MANQUANT: {TEXT_EMBEDDINGS_PATH}. Recherche texte désactivée.")
        
    logger.info("✅ CHARGEMENT BASE DE DONNÉES TERMINÉ.")
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
        # L'ajout de .reshape(1, -1) assure un format 1D si besoin, bien que flatten() soit déjà présent
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
        return {'success': False, 'error': 'Service de recherche texte non prêt (base de données ou modèle non chargé).'}, 503
        
    # 1. Extraire le vecteur de la requête utilisateur
    query_features = extract_text_features(query)
    if query_features is None:
        return {'success': False, 'error': 'Impossible de vectoriser la requête.'}, 500
    
    # 2. Calculer similarités (produit scalaire entre vecteurs normalisés = Cosine Similarity)
    logger.info("🧮 Calcul des similarités textuelles...")
        
    # Vérification de dimension (important pour éviter les crashes)
    if text_features_array.shape[1] != query_features.shape[0]:
        logger.error(f"Incompatibilité de dimension : DB {text_features_array.shape[1]} vs Query {query_features.shape[0]}")
        return {'success': False, 'error': 'Incompatibilité de dimension de vecteur.'}, 500
        
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
        
        # Le score de similarité est mis à l'échelle pour être plus lisible (ex: 0.895 -> 89.5%)
        scaled_similarity = round(score * 100, 2)
        
        top_results.append({
            'index': int(i),
            # La similarité est renvoyée en décimal (ex: 0.895) et non en pourcentage ici, le frontend fera la mise en forme.
            'similarity': float(score), 
            'nom': metadata.get('nom', 'N/A'),
            'artiste': metadata.get('artiste', 'N/A'),
            'annee': metadata.get('annee', 'N/A'),
            # Construction de l'URL relative. La route /images correspond à @app.route('/images/...')
            'image_url': f"/images/{metadata.get('image_id', '')}", 
            'lien_site': metadata.get('lien_site', 'N/A')
        })

    top_scores = [f'{s["similarity"]:.3f}' for s in top_results[:3]]
    logger.info("✅ Top 3 similarités texte: {}".format(top_scores))
    
    return top_results, 200

# --- INITIALISATION DE L'APPLICATION FLASK ---
# Assurez-vous que le chemin 'static' est correct pour vos images
app = Flask(__name__, static_folder='static')
CORS(app)

# Décorateur pour s'assurer que les BDD sont chargées au démarrage du serveur de production (Gunicorn)
@app.before_first_request
def setup_database_on_start():
    load_database()

# --- ROUTES D'API ---

# Route de vérification de l'état (Health Check)
@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie si l'API est en ligne et si le service de recherche texte est prêt."""
    status = 'ok'
    if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
        status = 'loading' # ou 'error' si l'erreur est fatale
        return jsonify({'status': status, 'message': 'API en ligne, base de données ou modèle en cours de chargement/non trouvé.'}), 503
    return jsonify({'status': status, 'message': 'API prête et service de recherche opérationnel.'}), 200


# Nouvelle route pour la recherche sémantique textuelle (via le chatbot)
# Cette route correspond au chemin dans le test PowerShell et dans le prompt vo.Dev.
@app.route('/api/search_text', methods=['POST', 'OPTIONS'])
def api_search_text():
    """ROUTE PRINCIPALE POUR LA RECHERCHE SÉMANTIQUE TEXTUELLE"""
    
    # Gestion du preflight CORS (important pour le frontend)
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response
            
    try:
        logger.info("🔍 API SEARCH TEXT ENDPOINT")
        
        # Vérification du service avant de commencer la logique
        if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
            return jsonify({'success': False, 'error': 'Service de recherche non initialisé.'}), 503

        data = request.json
        query = data.get('query')
        top_k = int(data.get('top_k', 5)) # Limité à 5 pour le frontend
        
        if not query:
            return jsonify({'success': False, 'error': 'Requête de recherche manquante.'}), 400
        
        # ⚠️ NOTE IMPORTANTE : La fonction search_logic_text renvoie maintenant une liste de résultats
        top_results, status_code = search_logic_text(query, top_k)
        
        # L'API doit retourner une liste, pas un dictionnaire comme dans la version précédente de votre code
        # On ne retourne plus de status_code interne à la logique, on retourne les résultats directement.
        return jsonify(top_results), status_code
            
    except Exception as e:
        logger.error(f"💥 API Search Text Error: {type(e).__name__}: {str(e)}")
        return jsonify({'success': False, 'error': 'Erreur interne du serveur. Voir logs.'}), 500

# Route pour servir les images (essentielle pour le frontend)
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sert les images stockées dans static/images/"""
    # Cloud Run est sensible aux chemins. On s'assure que le chemin est correct.
    return send_from_directory('static/images', filename)

# TODO: INSEREZ ICI VOTRE ROUTE DE RECHERCHE PAR IMAGE EXISTANTE (/api/search)
# @app.route('/api/search', methods=['POST', 'OPTIONS'])
# def api_search_image():
#    # ... Votre code existant pour la recherche par image ...
#    pass


# --- LANCEMENT DE L'APPLICATION (Pour le test local uniquement) ---
if __name__ == '__main__':
    # Le chargement est fait ici pour le développement local
    load_database() 
    app.run(debug=True, port=int(os.environ.get('PORT', 8080))) # Utilisation du port 8080 pour Cloud Run
