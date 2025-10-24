# VERSION FINALE ET NETTOYEE
import os
import logging
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import sys # Ajout pour un meilleur logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout,
                    force=True)
logger = logging.getLogger(__name__)

# --- VARIABLES GLOBALES (MISES À JOUR) ---
features_array = None
image_ids = None
metadata_list = None
IMAGE_EMBEDDINGS_PATH = 'data/embeddings.pkl' # Vous aviez un warning "manquant", c'est normal si vous ne l'utilisez pas
text_features_array = None
text_metadata_list = None
TEXT_EMBEDDING_MODEL = None

# --- MODIFICATION 1 ---
# On utilise le nouveau modèle plus performant
MODEL_NAME_FREE = "paraphrase-multilingual-mpnet-base-v2" 

# --- MODIFICATION 2 ---
# Le chemin dans le conteneur doit correspondre au nouveau modèle
LOCAL_MODEL_PATH = '/app/models/paraphrase-multilingual-mpnet-base-v2' 

# --- MODIFICATION 3 ---
# On pointe vers le nouveau fichier .pkl que vous allez générer
TEXT_EMBEDDINGS_PATH = 'data/text_embeddings_mpnet.pkl'


# --- FONCTION DE CHARGEMENT DE BASE DE DONNÉES ---
def load_database():
    global features_array, image_ids, metadata_list, text_features_array, text_metadata_list, TEXT_EMBEDDING_MODEL
    logger.info("🚀 DÉBUT CHARGEMENT BASE DE DONNÉES DEPUIS FICHIERS LOCAUX")

    if os.path.exists(TEXT_EMBEDDINGS_PATH):
        try:
            logger.info(f"📦 Chargement {TEXT_EMBEDDINGS_PATH}...")
            with open(TEXT_EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
                text_features_array = np.array(data['features'])
                text_metadata_list = data['metadata']
            logger.info(f"✅ Embeddings Texte chargés: {text_features_array.shape}")

            logger.info(f"🧠 Chargement du modèle depuis le chemin local: {LOCAL_MODEL_PATH}...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            if not os.path.exists(LOCAL_MODEL_PATH):
                logger.error(f"❌ ERREUR FATALE: Dossier modèle local non trouvé: {LOCAL_MODEL_PATH}")
                logger.error("Cela signifie qu'il y a un problème dans le Dockerfile (Étape 4: RUN python -c ...)")
                raise FileNotFoundError(f"Dossier modèle manquant: {LOCAL_MODEL_PATH}")

            TEXT_EMBEDDING_MODEL = SentenceTransformer(LOCAL_MODEL_PATH, device=device)
            logger.info(f"✅ Modèle de requête Texte chargé sur {device}.")

        except Exception as e:
            logger.error(f"❌ Erreur FATALE lors du chargement: {str(e)}", exc_info=True)
            text_features_array = None
            text_metadata_list = None
            TEXT_EMBEDDING_MODEL = None
    else:
        logger.error(f"⚠️ FICHIER TEXTUEL CRITIQUE MANQUANT: {TEXT_EMBEDDINGS_PATH}.")
        logger.error("Assurez-vous d'avoir exécuté 'generate_text_embeddings...' et poussé le .pkl sur GitHub.")


    if os.path.exists(IMAGE_EMBEDDINGS_PATH):
        logger.info(f"📦 Chargement {IMAGE_EMBEDDINGS_PATH}...")
    else:
        logger.warning(f"⚠️ FICHIER IMAGE MANQUANT: {IMAGE_EMBEDDINGS_PATH} (C'est OK si non utilisé)")

    logger.info("🏁 Fin du chargement de la base de données.")
    return True

# --- FONCTIONS DE RECHERCHE ---
def extract_text_features(text_query):
    global TEXT_EMBEDDING_MODEL
    if TEXT_EMBEDDING_MODEL is None:
        logger.error("💥 Extraction Texte impossible: Modèle non chargé.")
        return None
    try:
        features = TEXT_EMBEDDING_MODEL.encode(text_query, normalize_embeddings=True, convert_to_numpy=True)
        return features.flatten()
    except Exception as e:
        logger.error(f"💥 Extraction Texte Error: {str(e)}", exc_info=True)
        return None

def search_logic_text(query, top_k=10):
    global text_features_array, text_metadata_list
    logger.info(f"Recherche reçue pour: '{query}' (top_k={top_k})")
    
    if text_features_array is None or TEXT_EMBEDDING_MODEL is None:
        logger.error("Service non prêt (embeddings ou modèle non chargés).")
        return {'success': False, 'error': 'Service non prêt'}, 503
    
    query_features = extract_text_features(query)
    if query_features is None:
        logger.error("Vectorisation de la requête impossible.")
        return {'success': False, 'error': 'Vectorisation impossible'}, 500
    
    if text_features_array.shape[1] != query_features.shape[0]:
        logger.error(f"Incompatibilité de dimension : DB {text_features_array.shape[1]} vs Query {query_features.shape[0]}")
        return {'success': False, 'error': 'Incompatibilité de dimension'}, 500
    
    similarity_scores = np.dot(text_features_array, query_features)
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    top_results = []
    logger.info(f"Top 3 scores bruts: {[similarity_scores[i] for i in sorted_indices[:3]]}")
    
    for rank in range(min(top_k, len(sorted_indices))):
        i = sorted_indices[rank]
        score = similarity_scores[i]
        if i >= len(text_metadata_list):
            logger.warning(f"Index {i} hors limites, score {score}. On ignore.")
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
    logger.info(f"Recherche terminée, {len(top_results)} résultats trouvés.")
    return {'success': True, 'results': top_results}, 200

# --- INITIALISATION ET ROUTES FLASK ---
app = Flask(__name__, static_folder='static')
CORS(app)
load_database()

@app.route('/', methods=['GET'])
def home():
    logger.info("Appel de la route racine '/'")
    return jsonify({'status': 'Bienvenue sur le moteur de recherche (Modèle: mpnet)', 'db_ready': text_features_array is not None})

@app.route('/health', methods=['GET'])
def health_check():
    status = 'ok' if text_features_array is not None and TEXT_EMBEDDING_MODEL is not None else 'loading'
    return jsonify({'status': status}), 200 if status == 'ok' else 503

@app.route('/api/search_text', methods=['POST'])
def api_search_text():
    try:
        # Forcer la lecture en UTF-8 résout les problèmes de PowerShell
        data = request.get_json(force=True)
        query = data.get('query')
        if not query:
            logger.error("Requête invalide: 'query' manquant.")
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
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)