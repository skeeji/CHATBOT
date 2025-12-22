import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Chemins des fichiers
CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def run_generation():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Erreur : {CSV_PATH} est introuvable.")
        return

    print("--- 1. Chargement du catalogue CSV ---")
    df = pd.read_csv(CSV_PATH)
    
    # On prépare le texte que l'IA va "lire" pour comprendre l'objet
    df['text_for_ai'] = df.apply(lambda r: 
        f"Nom: {r.get('Nom luminaire','')}. Artiste: {r.get('Artiste / Dates','')}. "
        f"Catégorie: {r.get('Catégorie','')}. Matériaux: {r.get('Matériaux','')}. "
        f"Description: {r.get('Description','')}", axis=1)

    print("--- 2. Chargement du modèle IA (MPNet) ---")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    print("--- 3. Calcul des vecteurs (Embeddings) ---")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    print("--- 4. Sauvegarde des métadonnées complètes ---")
    metadata = []
    for _, row in df.iterrows():
        # ON ENREGISTRE TOUT : Si tu ajoutes une colonne au CSV, ajoute-la ici
        metadata.append({
            'image_id': str(row.get('Image luminaire (Nom du fichier)', '')),
            'nom': str(row.get('Nom luminaire', 'N/A')),
            'artiste': str(row.get('Artiste / Dates', 'N/A')),
            'annee': str(row.get('Année', 'N/A')),
            'categorie': str(row.get('Catégorie', 'N/A')),
            'description': str(row.get('Description', 'N/A')),
            'materiaux': str(row.get('Matériaux', 'N/A')),
            'dimensions': str(row.get('Dimensions', 'N/A')),
            'lien_site': str(row.get('Lien site marchand', '#'))
        })

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'features': np.array(embeddings, dtype='float32'), 'metadata': metadata}, f)
    
    print(f"✅ Terminé ! Le fichier {OUTPUT_FILE} contient {len(metadata)} produits avec toutes les colonnes.")

if __name__ == "__main__":
    run_generation()