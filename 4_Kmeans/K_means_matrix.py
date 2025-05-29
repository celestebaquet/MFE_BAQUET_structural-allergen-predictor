from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix


# Chargement des données
X_mean = np.load('Training/stacked_mean.npy')
X_max = np.load('Training/stacked_max.npy')
X_min = np.load('Training/stacked_min.npy')
X_norm = np.load('Training/stacked_norm.npy')
X_std = np.load('Training/stacked_std.npy')
X_median = np.load('Training/stacked_median.npy')
y_training = np.load('Training/labels_Training.npy')

liste_X = [X_mean, X_max, X_min, X_norm, X_std, X_median]
liste_name = ['Mean', 'Max', 'Min', 'Norm', 'Std', 'Median']
#K_means
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")


for i in range(6):
    #fig, ax = plt.subplots(figsize=(8, 6))  # Nouvelle figure
    X_np = liste_X[i]

    # K-means
    y_predicted = kmeans.fit_predict(X_np)

    #Confusion matrix
    cm = confusion_matrix(y_training, y_predicted, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Rappel (Sensibilite)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificite
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    bacc = (sensitivity + specificity) / 2  # Balanced Accuracy (BACC)
    mcc = (tp * tn - fp * fn) / ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5 # MCC
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0 # f1-score
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Precision positive predictive (PPV)
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Faux Positif Rate (FPR)

    print(f"Pour {liste_name[i]}:")
    print(f"tp = {tp}, fn = {fn}, tn = {tn}, fp = {fp}")
    print(f"sn = {sensitivity:.4f}, sp = {specificity:.4f}, acc = {accuracy:.4f}, bacc = {bacc:.4f}")
    print(f"mcc = {mcc:.4f}, F1 = {f1_score:.4f}, ppv = {ppv:.4f}, fpr = {fpr:.4f}\n")

