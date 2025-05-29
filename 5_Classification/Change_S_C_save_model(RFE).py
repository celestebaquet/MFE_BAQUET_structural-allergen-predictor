import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

def load_data(concat_method: str):
    """
    Charge les données d'entraînement et de test selon la méthode de concaténation.

    Args:
        concat_method (str): Méthode de concaténation à tester dans cette liste : ('mean', 'min', 'max', 'norm', 'median', 'std')
    
    Returns:
        X_training (np.array): données d'entrainement
        y_training (np.array): label des données d'entrainement
        X_test (np.array): données de test
        y_test (np.array): label des données de test
    """
    # Les chemins des fichiers changent en fonction de la méthode de concaténation
    X_training = np.load(f'Training/stacked_{concat_method}.npy')
    y_training = np.load(f'Training/labels_Training.npy')
    X_test = np.load(f'Ind1_F2/stacked_{concat_method}.npy')
    y_test = np.load(f'Ind1_F2/labels.npy')
    
    print(f"Data loaded with method {concat_method}:")
    print(f"Training data shape: {X_training.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_training, y_training, X_test, y_test

def select_features(X, selected_features):
    return X[:, selected_features]

def evaluate_model(X_training, y_training, X_test, y_test, k_values, selector, classifier, concat):
    """
    Évalue les performances du modèle pour différentes valeurs de k, avec les modèle de feature selection et de classification voulus.
    """
    # Initialisation des résultats
    results = []
    sensitivities = []
    specificities = []
    baccuracies = []
    f1_scores = []
    mccs = []

    #Initisialisation du meilleur modèle
    best_score = 2
    best_model = None
    best_k = 0
    best_i = 0

    # Si le dossier de sauvegarde n'existe pas, on le crée
    if not os.path.exists('models'):
        os.makedirs('models')

    for k in k_values:
        print(f"Calcul pour k = {k}")

        # Créer un objet RFE à chaque itération pour ré-entraîner la sélection de caractéristiques
        rfe = RFE(classifier, n_features_to_select=k, step=15)
        print("Entrainement de RFE")
        rfe.fit(X_training, y_training)
        print("Entrainement termine")

        # Sélectionner les meilleures caractéristiques selon le classement de RFE
        selected_features = np.where(rfe.support_)[0]

        # Créer un transformer pour la sélection des features
        feature_selector = FunctionTransformer(select_features, kw_args={'selected_features': selected_features}, validate=False)
        clf_rfe = make_pipeline(feature_selector, MinMaxScaler(), classifier)
        clf_rfe.fit(X_training, y_training)
        y_pred = clf_rfe.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calcul des métriques
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        bacc = (sensitivity + specificity) / 2
        f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        mcc = (tp * tn - fp * fn) / ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5

        # Calucl pour savoir si le modèle est meilleur que l'ancien:
        composite_score = np.sqrt((1-specificity)**2+(1-sensitivity)**2)

        # Sauvegarder le modèle si son score composite est le meilleur
        if composite_score < best_score:
            best_score = composite_score
            best_model = clf_rfe # Enregistrer le modèle correspondant à ce k
            best_k = k  # Mémoriser le meilleur k
            best_i = len(sensitivities)
            # Si un meilleur modèle est trouvé, sauvegarder ce modèle et supprimer les anciens
            model_filename = f"models/best_model_{selector.__name__}_{classifier.__class__.__name__}_{concat}_k{best_k}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"Le meilleur modele a ete sauvegarde sous : {model_filename}")

            # Supprimer les anciens fichiers de modèles qui ne sont plus les meilleurs
            for filename in os.listdir('models'):
                file_path = os.path.join('models', filename)
                if filename != f"best_model_{selector.__name__}_{classifier.__class__.__name__}_{concat}_k{best_k}.pkl" and filename.startswith(f"best_model_{selector.__name__}_{classifier.__class__.__name__}_{concat}_"):
                    os.remove(file_path)
                    print(f"Ancien modele supprime : {file_path}")

        # Ajout des résultats à la liste
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        baccuracies.append(bacc)
        f1_scores.append(f1_score)
        mccs.append(mcc)
        
        results.append({
            'k': k,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'bacc': bacc,
            'f1_score': f1_score,
            'mcc': mcc,
            'composite_score': composite_score 
        })
    
    return results, sensitivities, specificities, baccuracies, f1_scores, mccs, best_k, best_i

def plot_performance(k_values, sensitivities, specificities, baccuracies, f1_scores, mccs, k_max_perf, i_best, selector, classifier, concat_method, step):
    """
    Trace les performances du modèle pour différentes valeurs de k.
    """
    repository_name = f'{selector}_{classifier}'
    os.makedirs(repository_name, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, baccuracies, label="BACC", marker='o', color="#5e9eff")
    plt.plot(k_values, sensitivities, label="Sensitivity", marker='s', color="#7de1a1")
    plt.plot(k_values, specificities, label="Specificity", marker='^', color="#f5a2c9")
    plt.plot(k_values, f1_scores, label="F1-Score", marker='D', color="#a68cd9")
    plt.plot(k_values, mccs, label="MCC", marker='*', color="#ffb084")

    # Marque la performance maximale
    plt.axvline(x=k_max_perf, color="#FF4081", linestyle='-.', label=f'K at max perf: {k_max_perf}')
    
    # Paramètres de performance SEP
    SEP_bacc = 0.808
    SEP_sn = 0.640
    SEP_sp = 0.976
    SEP_f1 = 0.707
    SEP_mcc = 0.675
    plt.axhline(y=SEP_bacc, color="#5e9eff", linestyle='--', label="SEP BACC")
    plt.axhline(y=SEP_sn, color="#7de1a1", linestyle='--', label="SEP Sensitivity")
    plt.axhline(y=SEP_sp, color="#f5a2c9", linestyle='--', label="SEP Specificity")
    plt.axhline(y=SEP_f1, color="#a68cd9", linestyle='--', label="SEP F1-Score")
    plt.axhline(y=SEP_mcc, color="#ffb084", linestyle='--', label="SEP MCC")

    # Annotation du point avec la meilleure performance
    text = f"k = {k_max_perf}\nSensitivity = {sensitivities[i_best]:.3f}\nSpecificity = {specificities[i_best]:.3f}\nBACC = {baccuracies[i_best]:.3f}\nF1-score = {f1_scores[i_best]:.3f}\nMCC = {mccs[i_best]:.3f}"
    plt.annotate(
        text,
        xy=(k_max_perf, 0.05),
        xytext=(100 + 5, 0.1),
        arrowprops=dict(arrowstyle='->'),
        bbox=dict(boxstyle='round', fc='white', ec="#FF4081", lw=1)
    )
    
    plt.ylim(0, 1.0)
    plt.xlabel(F"Number of features selected (k, step = {step}) ")
    plt.ylabel("Score")
    plt.title(f"Model's Performance - {selector}/{classifier} - {concat_method}")
    plt.legend(ncol=2, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{repository_name}/{selector}_{classifier}_{concat_method}.png", dpi=300)
    plt.show()

    # 2Courbe ROC en fonction de k
    plt.figure(figsize=(8, 6))
    plt.scatter(specificities, sensitivities, color="#ff7f0e", label="Number of selected features", marker='+')

    # Annotation des k
    for i, k in enumerate(k_values):
        plt.annotate(str(k), (specificities[i], sensitivities[i]), textcoords="offset points", xytext=(5, -5), fontsize=8)

    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    #plt.ylim(0, 1)
    #plt.xlim(0, 1)
    plt.title(f"Sn/Sp Distance - {selector}/{classifier} - {concat_method}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{repository_name}/{selector}_{classifier}_{concat_method}(Dist).png", dpi=300)
    plt.show()

def save_results_to_csv(results, filename):
    """
    Sauvegarde les résultats dans un fichier CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main(concat_method, classifier, selector, str_classifier, str_selector, step=5, k_start=1):
    """
    Fonction principale pour exécuter le code avec des paramètres personnalisés.
    """
    # Parcourt la liste dans concat_method
    for concat in concat_method:
        filename = 'results_' + str_selector + '_' + str_classifier + '_' + concat + '.csv'
        # Charger les données
        X_training, y_training, X_test, y_test = load_data(concat)
    
        # Définir les valeurs de k
        k_end = X_training.shape[1] + 1
        k_values = list(range(k_start, k_end, step))
    
        # Évaluer les modèles
        results, sensitivities, specificities, baccuracies, f1_scores, mccs,k_max_perf, i_best = evaluate_model(
            X_training, y_training, X_test, y_test, k_values, selector, classifier, concat
        )

        # Tracer les résultats
        plot_performance(k_values, sensitivities, specificities, baccuracies, f1_scores, mccs, k_max_perf, i_best, str_selector, str_classifier, concat, step)

        # Sauvegarder les résultats dans un fichier CSV
        save_results_to_csv(results, filename)

if __name__ == '__main__':
    # Parameters
    # Stacking method: 'mean', 'max', 'min', 'norm', 'median', 'std'
    concat_method = ['mean', 'max', 'min', 'norm', 'median', 'std']
    # feature selection : f_classif, mutual_info_classif
    selector = RFE
    str_selector = 'RFE'
    # Classificator : LinearSVC(max_iter=5000), GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    classifier = LinearSVC(max_iter=5000)
    str_classifier = 'SVM'
    step = 15
    k_start = 16    
    
    # Exécution du script avec des paramètres personnalisés
    main(concat_method, classifier, selector, str_classifier, str_selector, step, k_start)
