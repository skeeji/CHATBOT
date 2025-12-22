import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Chargement de la base optimisée
PKL_PATH = 'data/text_embeddings_mpnet.pkl'
db = pickle.load(open(PKL_PATH, 'rb'))
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@app.route('/api/search_text', methods=['POST'])
def search():
    data = request.get_json(force=True)
    query = data.get('query', '').lower()
    
    # 1. Recherche IA
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(db['features'], query_vec)

    # 2. Logique de Boost "Style Gemini/ChatGPT"
    results = []
    for i in range(len(db['metadata'])):
        m = db['metadata'][i]
        s = float(scores[i])
        
        # Boost par Matériau (ex: "laiton")
        for mat in ["laiton", "bronze", "verre", "bois", "marbre", "métal", "acier"]:
            if mat in query and mat in m['materiaux'].lower(): s += 0.20
            
        # Boost par Catégorie (ex: "lampe à poser", "table")
        if any(w in query for w in ["poser", "table", "bureau", "chevet"]) and "poser" in m['categorie'].lower(): s += 0.25
        if any(w in query for w in ["suspension", "lustre", "plafonnier"]) and "suspension" in m['categorie'].lower(): s += 0.25
        
        # Boost par Époque (ex: "années 50", "1950")
        if "50" in query and "50" in m['annee']: s += 0.20
        if "60" in query and "60" in m['annee']: s += 0.20

        # Boost par Prix ("pas cher" < 600€)
        if "pas cher" in query and 0 < m['prix'] < 600: s += 0.35
        if "luxe" in query and m['prix'] > 5000: s += 0.30

        item = m.copy()
        item['similarity'] = s
        results.append(item)

    # Tri final par score boosté
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return jsonify({'success': True, 'results': results[:int(data.get('top_k', 12))]})

@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    for item in db['metadata']:
        if item['luminaireId'] == id: return jsonify({'success': True, 'found': True, 'metadata': item})
    return jsonify({'success': False, 'found': False}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)