import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Chargement au démarrage
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
PKL_PATH = 'data/text_embeddings_mpnet.pkl'

print("🚀 Chargement de la base de données...")
with open(PKL_PATH, 'rb') as f:
    db_data = pickle.load(f)
model = SentenceTransformer(MODEL_NAME)

@app.route('/api/search_text', methods=['POST'])
def search_text():
    try:
        req = request.get_json(force=True)
        query = req.get('query', '')
        top_k = int(req.get('top_k', 10))

        # Encodage de la recherche
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        
        # Calcul de similarité (NumPy est ultra-rapide ici)
        scores = np.dot(db_data['features'], query_vec)
        indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in indices:
            item = db_data['metadata'][i].copy() # On prend tout le dictionnaire
            item['similarity'] = float(scores[i])
            # L'image_url est construite dynamiquement pour l'orchestrateur
            item['image_url'] = f"/images/{item['image_id']}"
            results.append(item)
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)