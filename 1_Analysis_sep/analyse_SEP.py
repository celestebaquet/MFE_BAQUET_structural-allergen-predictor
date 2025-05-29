import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour charger les données
def load_data(file_path : str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV et ajoute deux colonnes :
    - `True_Label` : la classe réelle (1 si allergène, 0 sinon).
    - `Predicted_Label` : la classe prédite (1 si allergène, 0 sinon).

    Args:
        filepath (str): Chemin du fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données avec True_Label et Predicted_Label.
    """
    # Lecture du fichier csv en tant que dataframe avec pandas
    df = pd.read_csv(file_path) 
    # Création de la catégorie "True_Label" qui correspondond à la vraie classification de la protéine (1 si allergène, 0 sinon)
    df["True_Label"] = df["Name"].apply(lambda x: "1" if x.startswith("allergen") or x.startswith("Positive") else "0")
    # Création de la catégorie "Predicted_Label" qui correspondond à la classification de la protéine faite par l'outil (1 si allergène, 0 sinon)
    df["Predicted_Label"] = df["Class"].apply(lambda x: "1" if x.startswith("Allergen") else "0")
    # Afficher les premières lignes pour vérifier la bonne fonctionnalité de la fonction
    #print(df.head(5))
    return df

# Fonction pour calculer les métriques
def compute_metrics(df: pd.DataFrame):
    """
    Calcule les principales métriques de classification basées sur la matrice de confusion.

    Args:
        df (pd.DataFrame): DataFrame contenant au minimum les colonnes "True_Label", 
                           "Predicted_Label" et "Probability".

    Returns:
        tuple: Contient les métriques suivantes :
            - cm (np.ndarray): Matrice de confusion.
            - TN (int): Vrai négatif.
            - FP (int): Faux positif.
            - FN (int): Faux négatif.
            - TP (int): Vrai positif.
            - sensitivity (float): Sensibilité (recall).
            - specificity (float): Spécificité.
            - accuracy (float): Précision globale.
            - balanced_accuracy (float): Balanced Accuracy.
            - mcc (float): Coefficient de corrélation de Matthews.
            - f1_score (float): Score F1.
            - ppv (float): Précision positive prédictive.
            - fpr (float): Taux de faux positifs.
            - auc (float): Aire sous la courbe ROC.
    """
    labels = ["1", "0"] # 1 pour allergène et 0 pour non-allergène
    # Calcul de la matrice de confusion
    cm = confusion_matrix(df["True_Label"], df["Predicted_Label"], labels=labels)
    #On récupère les prédictions
    TP, FN, FP, TN, = cm.ravel() # C'est normal que ce soit inverse parce que sinon ça comprends pas bien qui est positif et qui est negatif...
    # Calcul des métriques : 
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Rappel (Sensibilite)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # Specificite
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Accuracy
    bacc = (sensitivity + specificity) / 2  # Balanced Accuracy (BACC)
    mcc = (TP * TN - FP * FN) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))**0.5 # MCC
    f1_score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0 # f1-score
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0  # Precision positive predictive (PPV)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0  # Faux Positif Rate (FPR)
    auc = roc_auc_score(df["True_Label"], df["Probability"])
    
    return cm, TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, auc

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(cm: np.ndarray, labels: list):
    """
    Affiche la matrice de confusion sous forme de carte thermique.

    Args:
        cm (np.ndarray): Matrice de confusion.
        labels (list): Liste des labels de classification ("Allergen", "Non-Allergen").
    """
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predictions")
    plt.ylabel("Vrais Labels")
    plt.title("Matrice de Confusion")
    plt.show()

# Fonction pour l'affichage du radar plot des métriques
def plot_radar_metrics(sensitivity, specificity, accuracy, bacc, mcc, f1_score, auc):
    labels = [
        "Recall (Sensibilité)",
        "Spécificité",
        "Accuracy",
        "MCC",
        "F1-Score",
        "BACC",
        "AUC"
    ]
    values = [sensitivity, specificity, accuracy, mcc, f1_score, bacc, auc]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='darkorange', linewidth=2)
    ax.fill(angles, values, color='darkorange', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Radar des métriques du modèle", size=14, pad=20)
    plt.tight_layout()
    plt.show()


# Fonction pour sauvegarder les métriques
def save_results(file_path: str, cm: np.ndarray, TN: int, FP: int, FN: int, TP: int, sensitivity: float, specificity: float, accuracy: float, bacc: float, mcc: float, f1_score: float, ppv: float, fpr: float, auc: float):
    """
    Enregistre les résultats des métriques de classification dans un fichier texte.

    Args:
        filepath (str): Chemin du fichier de sortie.
        cm (np.ndarray): Matrice de confusion.
        TN (int): Vrai négatif.
        FP (int): Faux positif.
        FN (int): Faux négatif.
        TP (int): Vrai positif.
        sensitivity (float): Sensibilité.
        specificity (float): Spécificité.
        accuracy (float): Précision globale.
        balanced_accuracy (float): Balanced Accuracy.
        mcc (float): MCC (Matthews Correlation Coefficient).
        f1_score (float): Score F1.
        ppv (float): Précision positive prédictive.
        fpr (float): Taux de faux positifs.
        auc (float): Aire sous la courbe ROC.
    """
    with open(file_path, "w") as f:
        # Matrice de Confusion
        f.write("Matrice de Confusion:\n")
        f.write("\tPredictions ->\tAllergen\tNon-Allergen\n")
        f.write(f"\tAllergen\t\t{cm[0,0]}\t{cm[0,1]}\n")
        f.write(f"\tNon-Allergen\t{cm[1,0]}\t{cm[1,1]}\n")
        f.write(f"\tVrais Labels\n\n")
        # Rapport de Classification
        f.write("Rapport de Classification:\n")
        f.write(f"Vrai Positif (TP): {TP}\n")
        f.write(f"Faux Positif (FP): {FP}\n")
        f.write(f"Faux Negatif (FN): {FN}\n")
        f.write(f"Vrai Negatif (TN): {TN}\n\n")
        f.write(f"Sensibilite (Recall): {sensitivity:.4f}\n")
        f.write(f"Specificite: {specificity:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
        f.write(f"Precision positive predictive (PPV): {ppv:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
        f.write(f"Balanced Accuracy (BACC): {bacc:.4f}\n")
        f.write(f"Taux de faux positifs (FPR): {fpr:.4f}\n")
        f.write(f"AUC (Air sous la courbe ROC): {auc:.4f}\n")
    print(f"Les resultats ont ete enregistres dans {file_path}")

def main(input_file : str, output_file : str):
    """
    Exécute l'évaluation de la classification :
    1. Chargement des données
    2. Calcul des métriques
    3. Affichage de la matrice de confusion
    4. Sauvegarde des résultats

    Args:
        input_file (str): Chemin du fichier CSV d'entrée.
        output_file (str): Chemin du fichier texte de sortie.
    """
    # 1. Chargement des données
    df = load_data(input_file)
    # 2. Calcul des métriques
    cm,  TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, auc = compute_metrics(df)
    # 3. Affichage du radar plot
    plot_radar_metrics(sensitivity, specificity, accuracy, bacc, mcc, f1_score, auc)
    # 4. Sauvegarde des résultats
    save_results(output_file, cm, TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, auc)



if __name__ == "__main__":
    # Fichier contenant les résultats
    input_file = "resultats_ind4_filtre2.csv"
    # Fichier de sortie
    output_file = "resultats_ind4_filtre2_v4.out"

    main(input_file, output_file)