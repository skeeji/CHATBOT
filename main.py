import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
PKL_PATH = 'data/text_embeddings_mpnet.pkl'

# Chargement unique au démarrage pour la rapidité
print("🚀 Chargement des données...")
with open(PKL_PATH, 'rb') as f:
    db = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

@app.route('/api/search_text', methods=['POST'])
def search():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        top_k = int(data.get('top_k', 10))

        # Encodage et Recherche
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(db['features'], query_vec)
        indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in indices:
            res = db['metadata'][i].copy()
            res['similarity'] = float(scores[i])
            results.append(res)
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))