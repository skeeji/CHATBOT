import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def run_generation():
    print("--- 1. Chargement du CSV ---")
    # On force la lecture en chaînes de caractères pour éviter les erreurs de type
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    
    # Création du texte riche pour l'IA (On combine tout pour la recherche)
    df['text_for_ai'] = df.apply(lambda r: 
        f"Nom: {r['Nom luminaire']}. Artiste: {r['Artiste / Dates']}. "
        f"Matériaux: {r['Matériaux']}. Description: {r['Description']}. "
        f"Catégorie: {r['Catégorie']}", axis=1)

    print("--- 2. Chargement du modèle IA ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    print("--- 3. Calcul des vecteurs (Embeddings) ---")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    print("--- 4. Stockage des métadonnées avec clés camelCase ---")
    metadata = []
    for _, row in df.iterrows():
        # On utilise les noms de clés attendus par votre frontend (v0.dev)
        metadata.append({
            'luminaireId': row['Image luminaire (Nom du fichier)'],
            'nom': row['Nom luminaire'],
            'artiste': row['Artiste / Dates'],
            'annee': row['Année'],
            'description': row['Description'],
            'materiaux': row['Matériaux'],
            'dimensions': row['Dimensions'],
            'categorie': row['Catégorie'],
            'lienSite': row['Lien site marchand'],
            'imageUrl': f"/images/{row['Image luminaire (Nom du fichier)']}"
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': np.array(embeddings, dtype='float32'), 'metadata': metadata}, f)
    
    print(f"✅ Terminé ! {len(metadata)} produits prêts.")

if __name__ == "__main__":
    run_generation()