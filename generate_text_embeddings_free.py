import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings

# Supprimer les avertissements inutiles dans le terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def clean_price(text):
    """Extrait proprement le prix numérique (ex: '1 200 €' -> 1200)"""
    if not text or str(text) == 'nan' or '€' not in str(text): return 0
    t = str(text).replace('\xa0', '').replace(' ', '').replace('€', '')
    nums = re.findall(r'\d+', t)
    return int(nums[0]) if nums else 0

def run():
    print("--- 🚀 DÉBUT DE LA GÉNÉRATION HAUTE PRÉCISION ---")
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    
    # Dictionnaire de synonymes pour aider l'IA
    syns = {
        "Lampe à poser": "lampe de table, bureau, chevet, luminaire à poser",
        "Suspension": "lustre, plafonnier, luminaire suspendu",
        "Lampadaire": "liseuse, lampe de sol, grande lampe"
    }

    print("--- 🧠 Enrichissement des données (Catégories, Matériaux, Époques) ---")
    def build_text(r):
        cat = r['Catégorie']
        s = syns.get(cat, "")
        # On répète les infos clés pour donner du poids (Boost sémantique)
        return (f"CATÉGORIE: {cat} {cat} {s}. MATÉRIAUX: {r['Matériaux']} {r['Matériaux']}. "
                f"ARTISTE: {r['Artiste / Dates']}. NOM: {r['Nom luminaire']}. "
                f"ANNÉE: {r['Année']}. DESCRIPTION: {r['Description']}")

    df['text_for_ai'] = df.apply(build_text, axis=1)
    df['prix_num'] = df['Estimation'].apply(clean_price)

    print("--- ⏳ Encodage IA (cela peut prendre 1-2 minutes) ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    metadata = []
    for i, row in df.iterrows():
        metadata.append({
            'luminaireId': row['Image luminaire (Nom du fichier)'],
            'nom': row['Nom luminaire'],
            'artiste': row['Artiste / Dates'],
            'annee': row['Année'],
            'description': row['Description'],
            'materiaux': row['Matériaux'],
            'dimensions': row['Dimensions'],
            'categorie': row['Catégorie'],
            'prix': int(row['prix_num']),
            'imageUrl': f"/images/{row['Image luminaire (Nom du fichier)']}"
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': np.array(embeddings, dtype='float32'), 'metadata': metadata}, f)
    
    print(f"\n✅ RÉUSSITE : {len(metadata)} produits encodés avec précision.")

if __name__ == "__main__":
    run()