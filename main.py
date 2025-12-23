import os, pickle, numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Chargement de la base
DB_PATH = 'data/text_embeddings_mpnet.pkl'
if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
    print(f"✅ Base chargée : {len(db['metadata'])} luminaires")
else:
    db = {'metadata': [], 'features': []}
    print("❌ Erreur : Pickle introuvable")

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@app.route('/api/search_text', methods=['POST'])
def search():
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
        
        # BOOST mot-clé exact (Artiste, Matériaux)
        q_words = query.split()
        for word in q_words:
            if len(word) > 3:
                if word in item['artiste'].lower(): score += 0.3
                if word in item['materiaux'].lower(): score += 0.2
                if word in item['nom'].lower(): score += 0.2

        item['similarity'] = score
        results.append(item)

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return jsonify({'success': True, 'results': results[:top_k]})

@app.route('/api/luminaires/<id>', methods=['GET'])
def get_details(id):
    # Fix pour l'ID avec ou sans .jpg
    search_id = id if id.endswith('.jpg') else f"{id}.jpg"
    for item in db['metadata']:
        if item.get('luminaireId') == search_id:
            return jsonify({'success': True, 'luminaire': item})
    return jsonify({'success': False}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)