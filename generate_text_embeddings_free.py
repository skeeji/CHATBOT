import pandas as pd
import numpy as np
import pickle
import os
import re
from sentence_transformers import SentenceTransformer

CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def extract_price(text):
    if not text or str(text) == 'nan': return 0
    # Nettoyage pour extraire le chiffre (ex: "1 200 €" -> 1200)
    nums = re.findall(r'\d+', str(text).replace(' ', ''))
    return int(nums[0]) if nums else 0

def run_generation():
    # Lecture forcée en chaînes de caractères
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    
    # On ignore les lignes sans nom d'image
    df = df[df['Image luminaire (Nom du fichier)'].str.contains('.jpg|.png', na=False, case=False)]

    print(f"--- 1. Analyse de {len(df)} produits ---")
    
    # BOOST SÉMANTIQUE : On répète les infos cruciales pour l'IA
    def build_text(r):
        return (f"CATEGORIE: {r['Catégorie']} {r['Catégorie']}. "
                f"MATERIAUX: {r['Matériaux']} {r['Matériaux']}. "
                f"NOM: {r['Nom luminaire']}. "
                f"EPOQUE: {r['Année']}. PRIX: {r['Estimation']}. "
                f"DESCRIPTION: {r['Description']}")

    df['text_for_ai'] = df.apply(build_text, axis=1)
    df['prix_num'] = df['Estimation'].apply(extract_price)

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    metadata = []
    for i, row in df.iterrows():
        img_id = row['Image luminaire (Nom du fichier)'].strip()
        metadata.append({
            'luminaireId': img_id, # On utilise le nom de fichier comme ID pivot
            'nom': row['Nom luminaire'] or "Luminaire",
            'artiste': row['Artiste / Dates'],
            'annee': row['Année'],
            'description': row['Description'],
            'materiaux': row['Matériaux'],
            'dimensions': row['Dimensions'],
            'categorie': row['Catégorie'],
            'prix': int(row['prix_num']),
            'imageUrl': f"https://image-similarity-api-590690354412.us-central1.run.app/images/{img_id}"
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': np.array(embeddings, dtype='float32'), 'metadata': metadata}, f)
    print("✅ Base de données .pkl générée avec succès.")

if __name__ == "__main__":
    run_generation()