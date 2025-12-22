import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

db = pickle.load(open('data/text_embeddings_mpnet.pkl', 'rb'))
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@app.route('/api/search_text', methods=['POST'])
def search():
    data = request.get_json(force=True)
    query = data.get('query', '').lower()
    top_k = int(data.get('top_k', 10))
    
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(db['features'], query_vec)
    
    results = []
    for i in range(len(db['metadata'])):
        item = db['metadata'][i].copy()
        score = float(scores[i])
        
        # BOOST STRICT : Années 50 (détecte 50, 1950, 50s)
        if "50" in query and ("50" in item['annee'] or "1950" in item['annee']):
            score += 0.3
            
        # BOOST STRICT : Laiton
        if "laiton" in query and "laiton" in item['materiaux'].lower():
            score += 0.3

        # BOOST STRICT : Lampe à poser
        if ("poser" in query or "table" in query) and "poser" in item['categorie'].lower():
            score += 0.2

        item['similarity'] = score
        results.append(item)

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return jsonify({'success': True, 'results': results[:top_k]})

@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    for item in db['metadata']:
        if item['luminaireId'] == id:
            return jsonify({'success': True, 'data': item, 'metadata': item})
    return jsonify({'success': False, 'error': 'Non trouvé'}), 404

if __name__ == '__main__': app.run(host='0.0.0.0', port=8080)