import os
import subprocess
from Bio import SeqIO  # Necessite Biopython : pip install biopython
import logging
import csv
import shutil

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_fasta(input_fasta: str, output_base_dir: str):
    """
    Separe un fichier FASTA contenant plusieurs sequences en plusieurs fichiers individuels,
    chacun dans le dossier de sortie specifie.

    Args:
        input_fasta (str): Chemin du fichier FASTA d'entree
        output_base_dir (str): Dossier où stocker les fichiers individuels
    
    Returns:
        list: Liste des chemins des fichiers FASTA crees
    """
    logging.info(f"Separation du fichier {input_fasta} en fichiers fasta dans {output_base_dir}")

    # Supprime le dossier s'il existe déjà
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)

    # Recrée le dossier
    os.makedirs(output_base_dir, exist_ok=True)
    fasta_files = []

    for record in SeqIO.parse(input_fasta, "fasta"):
        seq_id = record.id  # Recuperer l'ID de la sequence
        fasta_path = os.path.join(output_base_dir, f"{seq_id}.fasta")  # Nom du fichier FASTA
        
        # ecrire la sequence dans un fichier FASTA
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(f">{record.id}\n{record.seq}\n")
        
        fasta_files.append(fasta_path)  # Ajouter a la liste des fichiers crees

    logging.info(f"--> {len(fasta_files)} sequences enregistrees dans {output_base_dir}.")
    return fasta_files

def find_files(directory: str, pattern1=".npy", pattern2="rank_001"):
    """
    Recherche le nom des fichiers correspondant aux motifs en arguments.

    Args:
        directory (str): Dossier dans lequel on cherche le fichier.
        pattern1 (str) [default = '.npy']: Motif 1
        pattern2 (str) [default = "rank_001"]: Motif 2
    
    Returns:
        list: Liste des fichiers contenant les motifs dans leurs noms.
    """
    matching_files = [
        f for f in os.listdir(directory)
        if pattern1 in f and pattern2 in f
    ]
    return matching_files

def run_colabfold(fasta_files: list, output_results_dir: str, output_csv: str):
    """
    Execute ColabFold sur chaque fichier FASTA specifie et stocke les resultats dans des dossiers
    dedies dans le dossier de resultats.

    Args: 
        fasta_files (list): Liste des fichiers FASTA a traiter
        output_results_dir (str): Dossier où enregistrer les resultats
        output_csv (str): Chemin vers le fichier CSV de sortie
    """

    logging.info(f"Lancement des calculs d'embeddings... ")
    a = len(fasta_files) # Nombre total de séquences à traiter

    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['#','Remaining', 'Seq_id', 'embedding', 'File name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        # Écrire les en-têtes uniquement si le fichier est vide
        if os.stat(output_csv).st_size == 0:
            writer.writeheader()
        
        i = 0 # Compteur de séquences déjà traitées

        for fasta_file in fasta_files:
            i += 1
            logging.info(f"Managing file {i} out of {a}\n")
            logging.info(f"--> Traitement de {fasta_file} avec ColabFold...")

            seq_id = os.path.splitext(os.path.basename(fasta_file))[0]  # Extraire l'ID de la sequence (nom du fichier)
            result_dir = os.path.join(output_results_dir, seq_id)  # Creer un dossier pour les resultats de chaque sequence

            if not os.path.exists(result_dir) or not find_files(result_dir): 
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir, exist_ok=True)  # Creer le dossier de resultats pour cette sequence
                else : 
                    shutil.rmtree(result_dir)
                    os.makedirs(result_dir, exist_ok=True)  # Creer le dossier de resultats pour cette sequence
        
                # Commande ColabFold
                cmd = [ "colabfold_batch", fasta_file, result_dir, "--save-single-representations"]

                # Executer la commande
                subprocess.run(cmd, check=True)

            # Rechercher le nom du fichier .npy correpondant à l'embedding
            name_file = find_files(result_dir)

            if not name_file:
                logging.error(f"NA pas fichier .npy pour {seq_id}")
                writer.writerow({ '#': i, 'Remaining': a-i, 'Seq_id': seq_id, 'embedding': "Pas trouvé", 'File name': None})

            else: writer.writerow({ '#': i, 'Remaining': a-i, 'Seq_id': seq_id, 'embedding': "Done", 'File name': name_file})

            logging.info(f"-- --> Analyse terminee pour {os.path.basename(fasta_file)} !\n")

    logging.info("Toutes les sequences ont ete traitees avec ColabFold")


def main(input_fasta: str, output_base_dir: str, output_results_dir: str, output_csv: str):
    """
    Programme principal : 
    1. Separe les sequences FASTA;
    2. execute ColabFold pour chaque séquence;
    3. stocke les resultats dans un dossier spécifique par sequence.

    Args:
        - input_fasta (str): Chemin vers le fichier FASTA.
        - output_base_dir (str): Chemin vers le répertoire pour stocker les séquences en FASTA individuellement
        - output_results_dir (str): Chemin vers le repertoire pour stocker les résultats.
        - output_csv (str): Chemin vers le fichier CSV pour stocker le nom des fichiers de modèle idéal.
    """
    # 0 mettre le chemin vers colabfold --> Le copier coller dans le terminal avant de lancer le code!
    #subprocess.run("export PATH=$PATH:/opt/localcolabfold/localcolabfold/colabfold-conda/bin")
    
    # 1 Separer le fichier FASTA en plusieurs fichiers individuels
    fasta_files = split_fasta(input_fasta, output_base_dir)

    # 2️ Executer ColabFold sur chaque fichier FASTA et enregistrer les resultats dans les dossiers appropries
    run_colabfold(fasta_files, output_results_dir, output_csv)


if __name__ == "__main__":
    #Fichier contenant toutes les séquences à traiter
    input_fasta = "set_Ind2.fasta"  
    # Repertoire pour stocker les séquences individuellement
    output_base_dir = "Ind2/seq_output_Ind2"  
    # Repertoire pour stocker les résultats par séquence
    output_results_dir = "Ind2/colabfold_output_Ind2"  
    # Fichier CSV pour stocker le nom du modèle à utiliser et si il n'y a pas de résultats
    output_csv = "Training/sequence_embedding.csv"

    # Lancement de la fonction principale
    main(input_fasta, output_base_dir, output_results_dir, output_csv)

