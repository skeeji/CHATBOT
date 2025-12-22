import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def run_generation():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    print(f"--- Indexation de {len(df)} produits ---")
    
    # On construit une mémoire ultra-précise pour l'IA
    # On répète les mots clés pour qu'ils soient prioritaires
    df['text_for_ai'] = df.apply(lambda r: 
        f"CATÉGORIE: {r['Catégorie']} {r['Catégorie']}. "
        f"MATÉRIAUX: {r['Matériaux']} {r['Matériaux']}. "
        f"NOM: {r['Nom luminaire']}. "
        f"ANNÉE: {r['Année']} {r['Année']}. "
        f"DIMENSIONS: {r['Dimensions']}. "
        f"ÉTIQUETTES: {r['Etiquette']}. "
        f"DESCRIPTION: {r['Description']}", axis=1)

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'luminaireId': row['Image luminaire (Nom du fichier)'], # ID temporaire (Nom fichier)
            'nom': row['Nom luminaire'] or "Sans nom",
            'artiste': row['Artiste / Dates'],
            'annee': row['Année'],
            'description': row['Description'],
            'materiaux': row['Matériaux'],
            'dimensions': row['Dimensions'],
            'etiquette': row['Etiquette'],
            'categorie': row['Catégorie'],
            'imageUrl': f"https://image-similarity-api-590690354412.us-central1.run.app/images/{row['Image luminaire (Nom du fichier)']}"
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': np.array(embeddings, dtype='float32'), 'metadata': metadata}, f)
    print("✅ Base de données .pkl générée avec Matériaux, Dimensions et Étiquettes.")

if __name__ == "__main__": run_generation()