import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd

def load_data(set_name: str, level_filter: str):
    print(f"Chargement des donnees pour le set '{set_name}' avec le filtre '{level_filter}'...")
    X_test = np.load(f'{set_name}/{level_filter}/stacked_mean.npy')
    y_test = np.load(f'{set_name}/{level_filter}/labels.npy')
    print(f" - Données chargees : X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    return X_test, y_test

def prediction_threshold(model, X_test, y_test, threshold):
    # Predict with threshold
    print(f"Prediction avec un seuil de {threshold}...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(f" - Predictions realisees : {np.bincount(y_pred)} (negatifs/positifs)")
    return y_proba, y_pred

def computation_metric(y_test, y_pred):
    print("Calcul des metriques de performance...")
    TN, FP,FN, TP = confusion_matrix(y_test, y_pred).ravel() # C'est normal que ce soit inverse parce que sinon ça comprends pas bien qui est positif et qui est negatif...
    # Calcul des métriques : 
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Rappel (Sensibilite)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # Specificite
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Accuracy
    bacc = (sensitivity + specificity) / 2  # Balanced Accuracy (BACC)
    mcc = (TP * TN - FP * FN) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))**0.5 # MCC
    f1_score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0 # f1-score
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0  # Precision positive predictive (PPV)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0  # Faux Positif Rate (FPR)
    dist = np.sqrt((1-specificity)**2 + (1-sensitivity)**2)
    auc = roc_auc_score(y_test, y_pred)
    print(f" - Sensitivity : {sensitivity:.3f}, Specificity : {specificity:.3f}, Sn/Sp dist : {dist:.3f}")
    
    return TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, auc, dist

def main(model, sets, filters, output_csv, threshold):
    results = []
    print("Debut de l'evaluation du modele...\n")

    for set in sets:
        for filter in filters:
            X_test, y_test = load_data(set, filter)
            y_proba, y_pred = prediction_threshold(model, X_test, y_test, threshold)
            tn, fp, fn, tp, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, auc, dist = computation_metric(y_test, y_pred)

            results.append({
                "Set": set,
                "Filter": filter,
                "Threshold": threshold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "Accuracy": accuracy,
                "Balanced Accuracy": bacc,
                "MCC": mcc,
                "F1 Score": f1_score,
                "PPV": ppv,
                "FPR": fpr,
                "AUC": auc,
                "Distance": dist
            })

            print(f"--- Fin de l'evaluation pour {set}-{filter} ---\n")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultats enregistres dans {output_csv}")
     

if __name__ == "__main__":
    #Chemin vers le modèle à tester
    model_path = 'best_model_f_classif_XGBClassifier_mean_k241.pkl'
    #Load du modèle avec pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    #Liste des set à tester
    sets = ["Ind1", "Ind2", "Ind3", "Ind4"]
    #Liste des niveaux de filtrages
    filters = ["Original", "F1", "F2"]

    #Threshold
    threshold = 0.3

    #Chemin vers le CSV dans lequel les résultats sont sauvegardés.
    output_csv = "results_modèle_final_0.3.csv"

    main(model, sets, filters, output_csv, threshold)
    