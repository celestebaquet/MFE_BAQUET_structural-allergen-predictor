from Bio import SeqIO
import os
import shutil

def get_fasta_ids(fasta_file: str) -> set[str]:
    """
    Extrait les identifiants des sequences d'un fichier FASTA.

    Args:
        fasta_file (str): Chemin du fichier FASTA.

    Returns:
        set[str]: Ensemble des identifiants trouves dans le fichier FASTA.
    """
    return {record.id for record in SeqIO.parse(fasta_file, "fasta")}

input_fasta = "Ind3/sep_ind3_40_cdhit2.fasta"

# Liste des fichiers (sans chemin, avec ou sans .npy selon ton besoin)
fichiers_voulus = get_fasta_ids(input_fasta)

# Chemins
dossier_source = 'Ind3\NPY_Ind3'
dossier_destination = 'Ind3\NPY_Ind3_filtre2'

# Création du dossier de destination s'il n'existe pas
os.makedirs(dossier_destination, exist_ok=True)

# Parcours des fichiers voulus
for base_name in fichiers_voulus:
    nom_fichier = base_name + '.npy'
    chemin_source = os.path.join(dossier_source, nom_fichier)
    chemin_dest = os.path.join(dossier_destination, nom_fichier)
    
    # Vérifie que le fichier existe avant de copier
    if os.path.isfile(chemin_source):
        shutil.copy2(chemin_source, chemin_dest)
        print(f'Copie : {nom_fichier}')
    else:
        print(f'Fichier introuvable : {nom_fichier}')