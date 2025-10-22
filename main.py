# VERSION FINALE ET NETTOYEE
import os
import logging
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

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

    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH}...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte Gratuits chargés: {text_features_array.shape}")

            logger.info(f"🧠 Chargement du modèle {MODEL_NAME_FREE}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            TEXT_EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME_FREE, device=device)
            logger.info(f"✅ Modèle de requête Texte chargé sur {device}.")

        except Exception as e:
            logger.error(f"❌ Erreur FATALE lors du chargement: {str(e)}", exc_info=True)
            # Met les variables à None pour que le health check échoue
            text_features_array = None
            text_metadata_list = None
            TEXT_EMBEDDING_MODEL = None
    else:
        logger.error(f"⚠️ FICHIER TEXTUEL CRITIQUE MANQUANT: {TEXT_EMBEDDINGS_PATH}.")

    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH}...")
    else:
        logger.warning(f"⚠️ FICHIER IMAGE MANQUANT: {IMAGE_EMBEDDINGS_PATH}")

    logger.info("🏁 Fin du chargement de la base de données.")
    return True

# --- FONCTIONS DE RECHERCHE ---
def extract_text_features(text_query):
    global TEXT_EMBEDDING_MODEL
    if TEXT_EMBEDDING_MODEL is None: return None
    try:
        features = TEXT_EMBEDDING_MODEL.encode(text_query, normalize_embeddings=True, convert_to_numpy=True)
        return features.flatten()
    except Exception as e:
        logger.error(f"💥 Extraction Texte Error: {str(e)}")
        return None

def search_logic_text(query, top_k=10):
    global text_features_array, text_metadata_list
    if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
        return {'success': False, 'error': 'Service non prêt'}, 503
    
    query_features = extract_text_features(query)
    if query_features is None:
        return {'success': False, 'error': 'Vectorisation impossible'}, 500
    
    if text_features_array.shape[1] != query_features.shape[0]:
        logger.error(f"Incompatibilité de dimension : DB {text_features_array.shape[1]} vs Query {query_features.shape[0]}")
        return {'success': False, 'error': 'Incompatibilité de dimension'}, 500
    
    similarity_scores = np.dot(text_features_array, query_features)
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    top_results = []
    for rank in range(min(top_k, len(sorted_indices))):
        i = sorted_indices[rank]
        score = similarity_scores[i]
        if i >= len(text_metadata_list): continue
        metadata = text_metadata_list[i]
        top_results.append({
            'index': int(i), 'similarity': float(score),
            'nom': metadata.get('nom', 'N/A'), 'artiste': metadata.get('artiste', 'N/A'),
            'annee': metadata.get('annee', 'N/A'), 'image_url': f"/images/{metadata.get('image_id', '')}",
            'lien_site': metadata.get('lien_site', 'N/A')
        })
    return {'success': True, 'results': top_results}, 200

# --- INITIALISATION ET ROUTES FLASK ---
app = Flask(__name__, static_folder='static')
CORS(app)
load_database()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Bienvenue sur la version corrigée', 'db_ready': text_features_array is not None})

@app.route('/health', methods=['GET'])
def health_check():
    status = 'ok' if text_features_array is not None and TEXT_EMBEDDING_MODEL is not None else 'loading'
    return jsonify({'status': status}), 200 if status == 'ok' else 503

@app.route('/api/search_text', methods=['POST'])
def api_search_text():
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'Requête manquante'}), 400
        
        result, status_code = search_logic_text(query, int(data.get('top_k', 10)))
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"💥 API Search Text Error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Erreur interne'}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

