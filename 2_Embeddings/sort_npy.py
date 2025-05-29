import os
import numpy as np
import pandas as pd
import shutil

def move_directory(csv_file: str, input_directory: str, output_directory: str):
    """
    1. Lis le fichier CSV
    2. Cherche le fichier correspondant
    3. Le copie dans un nouveau dossier et change le nom (le simplifie)

    Args:
        csv_file (str): Chemin vers le fichier CSV.
        input_directory (str): Chemin vers les dossiers à chercher.
        output_directory (str): Dossier pour stocker les fichiers.
    """
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)

    # Parcourir les lignes du CSV
    for index, row in df.iterrows():
        # Nom du fichier .npy à partir de la cinquième colonne (index 4)
        input_filename = row[4]
        input_filename = input_filename.strip("[]").replace("'", "")
        # Nouveau nom à partir de la troisième colonne (index 2)
        filename = row[2]
        new_filename = row[2] + '.npy'
    
        # Chemin complet du fichier source
        input_file_path = os.path.join(input_directory, filename, input_filename)
    
        # Vérifier si le fichier .npy existe dans le répertoire d'entrée
        if os.path.exists(input_file_path):
            # Chemin de destination avec le nouveau nom
            output_file_path = os.path.join(output_directory, new_filename)
        
            # Renommer (copier) le fichier dans le répertoire de sortie
            shutil.copy(input_file_path, output_file_path)
            print(f"Fichier {input_filename} renommé en {new_filename}")
        else:
            print(f"Le fichier {input_filename} n'existe pas dans le répertoire d'entrée.")


if __name__ == "__main__":
    # Chemin du fichier CSV
    csv_file = 'sequence_embedding.csv'
    # Répertoire contenant les fichiers .npy
    input_directory = 'colabfold_output_Ind2'
    # Répertoire où les fichiers renommés seront stockés
    output_directory = 'file_npy_Ind2'

    # Assurer que le répertoire de sortie existe
    os.makedirs(output_directory, exist_ok=True)
    
    move_directory(csv_file, input_directory, output_directory)
    



