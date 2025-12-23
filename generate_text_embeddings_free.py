import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

CSV_PATH = "luminaires_export_2025-08-28 (4).csv"
OUTPUT_FILE = "data/text_embeddings_mpnet.pkl"

def run_generation():
    if not os.path.exists('data'): os.makedirs('data')
    
    # Chargement du CSV
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    print(f"--- Indexation de {len(df)} luminaires avec focus descriptif ---")
    
    # LOGIQUE DE CONSTRUCTION DU "CERVEAU"
    def create_rich_text(r):
        # 1. On commence par les attributs visuels forts (Etiquette contient souvent couleur/style)
        # On les répète pour donner plus de "poids" statistique
        visuel = f"COULEURS ET STYLE: {r['Etiquette']} {r['Etiquette']}. "
        
        # 2. On ajoute l'artiste et le nom (essentiel pour l'identité)
        identite = f"ARTISTE: {r['Artiste / Dates']}. NOM: {r['Nom luminaire']}. "
        
        # 3. Les matériaux et la catégorie
        technique = f"TYPE: {r['Catégorie']}. MATÉRIAUX: {r['Matériaux']}. "
        
        # 4. La description (qui contient les formes : "ovale", "organique", "géométrique")
        description = f"DESCRIPTION DÉTAILLÉE: {r['Description']}. "
        
        # On assemble le tout. L'ordre et la répétition indiquent à l'IA ce qui est le plus important.
        return visuel + identite + technique + description

    print("Construction des descriptions enrichies...")
    df['text_for_ai'] = df.apply(create_rich_text, axis=1)

    # Chargement du modèle multilingue
    print("Chargement du modèle SentenceTransformer...")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    
    # Génération des embeddings (vecteurs mathématiques)
    print("Génération des vecteurs (cette étape peut prendre quelques minutes)...")
    embeddings = model.encode(df['text_for_ai'].tolist(), show_progress_bar=True, normalize_embeddings=True)

    # Préparation des métadonnées pour le frontend
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'luminaireId': row['Image luminaire (Nom du fichier)'],
            'nom': row['Nom luminaire'] if row['Nom luminaire'] else "Sans nom",
            'artiste': row['Artiste / Dates'],
            'annee': row['Année'],
            'description': row['Description'],
            'materiaux': row['Matériaux'],
            'dimensions': row['Dimensions'],
            'etiquette': row['Etiquette'],
            'categorie': row['Catégorie'],
            'imageUrl': f"https://image-similarity-api-590690354412.us-central1.run.app/images/{row['Image luminaire (Nom du fichier)']}"
        })

    # Sauvegarde du fichier de "cerveau"
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({'metadata': metadata, 'features': embeddings}, f)
    
    print(f"✅ Succès ! Le fichier {OUTPUT_FILE} est prêt pour le chatbot.")

if __name__ == "__main__":
    run_generation()