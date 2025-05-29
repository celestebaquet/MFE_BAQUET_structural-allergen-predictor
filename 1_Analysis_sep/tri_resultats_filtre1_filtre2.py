from Bio import SeqIO
import csv

def get_fasta_ids(fasta_file: str) -> set[str]:
    """
    Extrait les identifiants des sequences d'un fichier FASTA.

    Args:
        fasta_file (str): Chemin du fichier FASTA.

    Returns:
        set[str]: Ensemble des identifiants trouves dans le fichier FASTA.
    """
    return {record.id for record in SeqIO.parse(fasta_file, "fasta")}

def compare_fasta(file1: str, file2: str) -> set[str]:
    """
    Compare deux fichiers FASTA et retourne les identifiants uniques au premier fichier.

    Args:
        file1 (str): Chemin du premier fichier FASTA.
        file2 (str): Chemin du deuxieme fichier FASTA.

    Returns:
        set[str]: Identifiants presents dans file1 mais absents de file2.
    """
    ids1 = get_fasta_ids(file1)
    ids2 = get_fasta_ids(file2)
    
    unique_ids = ids1 - ids2  # Identifiants presents dans file1 mais pas dans file2
    return unique_ids

def filter_csv(input_csv: str, output_csv: str, unique_ids: set[str]):
    """
    Supprime du fichier CSV les lignes dont l'identifiant est dans unique_ids.

    Args:
        input_csv (str): Chemin du fichier CSV à filtrer.
        output_csv (str): Chemin du fichier CSV filtre en sortie.
        unique_ids (set[str]): Ensemble des identifiants à exclure.

    Returns:
        None. Le fichier filtre est sauvegarde sous output_csv.
    """
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # Lire l'en-tête
        writer.writerow(header)  # Écrire l'en-tête dans le nouveau fichier

        for row in reader:
            seq_id = row[0]  # L'identifiant est dans la premiere colonne
            if seq_id not in unique_ids:  # Garder seulement ceux qui ne sont PAS dans unique_ids
                writer.writerow(row)

if __name__ == "__main__":
    # Definition des fichiers en entree et sortie
    input_csv = 'resultats_ind4.csv'  # Fichier CSV à filtrer
    output_csv = 'resultats_ind4_filtre2.csv'  # Fichier de sortie apres filtrage
    fasta_1 = 'ind_4_40.fasta'  # Premier fichier FASTA
    fasta_2 = 'sep_ind4_40_cdhit2.fasta'  # Deuxieme fichier FASTA (corrige du 'ind1...')

    # Comparaison des fichiers FASTA pour obtenir les identifiants uniques
    unique_ids = compare_fasta(fasta_1, fasta_2)

    # Filtrage du fichier CSV en supprimant les sequences trouvees dans fasta_2
    filter_csv(input_csv, output_csv, unique_ids)
