import os, pickle, numpy as np, logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Configuration logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION CHEMINS ---
# Ce chemin doit correspondre EXACTEMENT à celui du Dockerfile
MODEL_LOCAL_PATH = '/app/models/paraphrase-multilingual-mpnet-base-v2'
DB_PATH = 'data/text_embeddings_mpnet.pkl'

# Chargement de la base de données
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
    logger.info(f"✅ Base chargée : {len(db['metadata'])} luminaires")
else:
    db = {'metadata': [], 'features': []}
    logger.error("❌ Erreur : Fichier Pickle (DB) introuvable")

# Chargement du modèle (Local ou téléchargé si local absent)
try:
    if os.path.exists(MODEL_LOCAL_PATH):
        logger.info("📦 Chargement du modèle depuis le stockage LOCAL du conteneur...")
        model = SentenceTransformer(MODEL_LOCAL_PATH)
    else:
        logger.warning("🌐 Modèle local absent, tentative de téléchargement distant...")
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
except Exception as e:
    logger.error(f"❌ Erreur critique chargement modèle : {e}")
    model = None

@app.route('/api/search_text', methods=['POST'])
def search():
    if model is None:
        return jsonify({'success': False, 'error': 'Modèle non disponible'}), 500
        
    data = request.get_json(force=True)
    query = data.get('query', '').lower()
    top_k = int(data.get('top_k', 40))
    
    if not query: return jsonify({'success': True, 'results': []})

    # Recherche vectorielle
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(db['features'], query_vec)
    
    results = []
    for i in range(len(db['metadata'])):
        item = db['metadata'][i].copy()
        score = float(scores[i])
        
        # BOOST mot-clé exact
        q_words = query.split()
        for word in q_words:
            if len(word) > 3:
                if word in item.get('artiste', '').lower(): score += 0.3
                if word in item.get('materiaux', '').lower(): score += 0.2
                if word in item.get('nom', '').lower(): score += 0.2

        item['similarity'] = score
        results.append(item)

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return jsonify({'success': True, 'results': results[:top_k]})

@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    search_id = id if id.endswith('.jpg') else f"{id}.jpg"
    for item in db['metadata']:
        if item.get('luminaireId') == search_id:
            return jsonify({'success': True, 'luminaire': item})
    return jsonify({'success': False}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
