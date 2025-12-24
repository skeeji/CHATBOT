import os, pickle, numpy as np, logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

# Configuration des Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = "/app/models/paraphrase-multilingual-mpnet-base-v2"
DB_PATH = "data/text_embeddings_mpnet.pkl"

# 1. Chargement de la base Pickle
db = {'metadata': [], 'features': []}
if os.path.exists(DB_PATH):
    try:
        with open(DB_PATH, 'rb') as f:
            db = pickle.load(f)
        logger.info(f"✅ Base chargée : {len(db['metadata'])} luminaires")
    except Exception as e:
        logger.error(f"❌ Erreur lecture Pickle : {e}")
else:
    logger.error(f"❌ Fichier {DB_PATH} introuvable !")

# 2. Chargement du modèle (Local)
logger.info("⏳ Chargement du modèle SentenceTransformer...")
try:
    # On force le chargement depuis le dossier local créé au build Docker
    model = SentenceTransformer(MODEL_PATH)
    logger.info("✅ Modèle chargé avec succès depuis le stockage local.")
except Exception as e:
    logger.error(f"❌ Erreur chargement modèle : {e}")
    model = None

@app.route('/api/search_text', methods=['POST'])
def search():
    if model is None:
        return jsonify({"success": False, "error": "Modèle non chargé"}), 500

    try:
        data = request.get_json(force=True)
        query = data.get('query', '').lower()
        top_k = int(data.get('top_k', 40))
        
        logger.info(f"🔍 Recherche texte : '{query}'")
        
        if not query:
            return jsonify({'success': True, 'results': []})

        # Encodage et calcul de similarité vectorielle
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(db['features'], query_vec)
        
        results = []
        for i in range(len(db['metadata'])):
            item = db['metadata'][i].copy()
            score = float(scores[i])
            
            # Boost textuel direct
            q_words = query.split()
            for word in q_words:
                if len(word) > 3:
                    if word in item.get('nom', '').lower(): score += 0.2
                    if word in item.get('artiste', '').lower(): score += 0.3
            
            item['similarity'] = score
            results.append(item)

        # Tri et renvoi du top_k
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)
        return jsonify({'success': True, 'results': results[:top_k]})
    except Exception as e:
        logger.error(f"💥 Erreur : {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    search_id = id if id.endswith('.jpg') else f"{id}.jpg"
    for item in db['metadata']:
        if item.get('luminaireId') == search_id:
            return jsonify({'success': True, 'luminaire': item})
    return jsonify({'success': False}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
