import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch 

# --- CONFIGURATION ---
CSV_PATH = "luminaires_export_2025-08-28 (4) - Bien.csv" 
OUTPUT_DIR = "data"

# --- MODIFICATION 1 ---
# On change le nom du modèle pour un plus performant et multilingue
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 

# --- MODIFICATION 2 ---
# On change le nom du fichier de sortie pour correspondre au nouveau modèle
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "text_embeddings_mpnet.pkl") 


# --- PROCESSUS PRINCIPAL ---
def process_catalogue():
    print("Chargement du catalogue...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERREUR: Fichier CSV non trouvé à: {CSV_PATH}")
        return

    # 1. Créer une colonne de description riche
    print("Préparation des descriptions sémantiques...")
    df['semantic_description'] = df.apply(lambda row: 
        f"Nom: {row.get('Nom luminaire', 'Inconnu') if pd.notna(row.get('Nom luminaire')) else 'Inconnu'}. "
        f"Artiste: {row.get('Artiste / Dates', 'Inconnu') if pd.notna(row.get('Artiste / Dates')) else 'Inconnu'}. "
        f"Année: {row.get('Année', 'Inconnue') if pd.notna(row.get('Année')) else 'Inconnue'}. "
        f"Catégorie: {row.get('Catégorie', 'Inconnue') if pd.notna(row.get('Catégorie')) else 'Inconnue'}. "
        f"Matériaux: {row.get('Matériaux', 'Non spécifié') if pd.notna(row.get('Matériaux')) else 'Non spécifié'}. "
        f"Mots-clés: {row.get('Etiquette', 'Aucun') if pd.notna(row.get('Etiquette')) else 'Aucun'}. "
        f"Description détaillée: {row.get('Description', '') if pd.notna(row.get('Description')) else ''}", 
        axis=1
    )
    
    descriptions = df['semantic_description'].tolist()
    
    # 2. Charger le nouveau modèle Sentence Transformer
    print(f"Chargement du nouveau modèle d'embedding: {MODEL_NAME}...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        model = SentenceTransformer(MODEL_NAME, device=device)
        print(f"Modèle chargé sur {device}.")
    except Exception as e:
        print(f"Erreur de chargement du modèle: {e}.")
        return

    # 3. Générer les embeddings
    print(f"Génération de {len(descriptions)} embeddings (cela peut prendre du temps)...")
    embeddings = model.encode(
        descriptions, 
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # 4. Préparer les métadonnées
    metadata = []
    print("Préparation des métadonnées...")
    for index, row in df.iterrows():
        metadata.append({
            'image_id': row.get('Image luminaire (Nom du fichier)', ''),
            'nom': row.get('Nom luminaire', 'N/A'),
            'artiste': row.get('Artiste / Dates', 'N/A'),
            'annee': row.get('Année', 'N/A'),
            'lien_site': row.get('Lien site marchand', '#') 
        })
            
    # 5. Sauvegarder les résultats
    if embeddings.shape[0] > 0:
        embeddings_array = np.array(embeddings, dtype='float32')
        data_to_save = {
            'features': embeddings_array,
            'metadata': metadata
        }
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        print(f"\n✅ Terminé! {embeddings.shape[0]} embeddings et métadonnées sauvegardés dans '{OUTPUT_FILE}'.")
    else:
        print("\n❌ Échec de la génération des embeddings.")

if __name__ == "__main__":
    print("\n-------------------------------------------------------------")
    print("--- ATTENTION : Ceci est un long processus de génération ! ---")
    print("-------------------------------------------------------------\n")
    process_catalogue()