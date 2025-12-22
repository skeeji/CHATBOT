import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Chargement de la base
PKL_PATH = 'data/text_embeddings_mpnet.pkl'
db = pickle.load(open(PKL_PATH, 'rb'))
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@app.route('/api/search_text', methods=['POST'])
def search():
    data = request.get_json(force=True)
    query = data.get('query', '').lower()
    top_k = int(data.get('top_k', 10))

    # 1. Recherche IA
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(db['features'], query_vec)

    # 2. Logique de Boost "Style ChatGPT/Gemini"
    results = []
    for i in range(len(db['metadata'])):
        m = db['metadata'][i]
        s = float(scores[i])
        
        # Boost Matériaux (Laiton, etc.)
        if any(w in query for w in ["laiton", "bronze", "verre"]) and any(w in query for w in m['materiaux'].lower().split()):
            s += 0.25
        
        # Boost Années 50
        if "50" in query and "50" in m['annee']: s += 0.20
        
        # Boost Prix (Pas cher < 600€)
        if "pas cher" in query and 0 < m['prix'] < 600: s += 0.30

        item = m.copy()
        item['similarity'] = s
        results.append(item)

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return jsonify({'success': True, 'results': results[:top_k]})

# CETTE ROUTE RÉPARE L'ERREUR "Luminaire non trouvé"
@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    for item in db['metadata']:
        if item['luminaireId'] == id:
            return jsonify({'success': True, 'data': item, 'metadata': item})
    return jsonify({'success': False, 'error': 'Non trouvé dans le CSV'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)