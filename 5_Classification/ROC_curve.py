import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def select_features(X, selected_features):
    return X[:, selected_features]

def load_data(concat_method: str):
    X_train = np.load(f'Training/stacked_{concat_method}.npy')
    y_train = np.load(f'Training/labels_Training.npy')
    X_test = np.load(f'Ind1_F2/stacked_{concat_method}.npy')
    y_test = np.load(f'Ind1_F2/labels.npy')
    return X_train, X_test, y_train, y_test


# Model configuration
#models = {
#    'GB_Mutual_info_median': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_median_k196.pkl', 'median', 'plum'),
#    'GB_Mutual_info_mean': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_mean_k121.pkl', 'mean', 'darkorchid'),
#    'GB_Mutual_info_norm': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_norm_k256.pkl', 'norm', 'powderblue'),
#    'GB_Mutual_info_max': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_max_k61.pkl', 'max', 'yellowgreen'),
#    'GB_Mutual_info_min': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_min_k226.pkl', 'min', 'salmon'),
#    'GB_Mutual_info_std': ('mutual_info_classif_GB/best_model_mutual_info_classif_GradientBoostingClassifier_std_k181.pkl', 'std', 'indianred')
#}

models = {
    'XGB_RFE_median': ('RFE_XGBoost/best_model_RFE_XGBClassifier_median_k241.pkl', 'median', 'plum'),
    'XGB_f_classif_mean': ('f_classif_XGBoost/best_model_f_classif_XGBClassifier_mean_k241.pkl', 'mean', 'darkorchid')
    #'XGB_mutual_info_classif_mean': ('mutual_info_classif_XGBoost/best_model_mutual_info_classif_XGBClassifier_mean_k136.pkl', 'mean', 'darkorchid'),
    #'XGB_mutual_info_classif_norm': ('mutual_info_classif_XGBoost/best_model_mutual_info_classif_XGBClassifier_norm_k256.pkl', 'norm', 'powderblue'),
    #'XGB_mutual_info_classif_max': ('mutual_info_classif_XGBoost/best_model_mutual_info_classif_XGBClassifier_max_k166.pkl', 'max', 'yellowgreen'),
    #'XGB_mutual_info_classif_min': ('mutual_info_classif_XGBoost/best_model_mutual_info_classif_XGBClassifier_min_k256.pkl', 'min', 'salmon'),
    #'XGB_mutual_info_classif_std': ('mutual_info_classif_XGBoost/best_model_mutual_info_classif_XGBClassifier_std_k226.pkl', 'std', 'indianred')
}

# ------------------ ROC Curve Figure ------------------
plt.figure(figsize=(12, 6))
for label, (model_path, method, color) in models.items():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_train, X_test, y_train, y_test = load_data(method)
    y_test_score = model.predict_proba(X_test)[:, 1]
    y_train_score = model.predict_proba(X_train)[:, 1]

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_score)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_score)
    auc_score_test = auc(fpr_test, tpr_test)
    auc_score_train = auc(fpr_train, tpr_train)

    plt.plot(fpr_test, tpr_test, lw=2, label=f'{label} (test) AUC={auc_score_test:.4f}', color=color)
    plt.plot(fpr_train, tpr_train, lw=2, linestyle='--', label=f'{label} (train) AUC={auc_score_train:.4f}', color=color)
    

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
plt.title('ROC Curves per GB Classifier (Training + Ind1 (F2) Sets)', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(fontsize=12)
plt.tight_layout()
output_path = os.path.join('mutual_info_classif_XGBoost', f"ROC.png")
#plt.savefig(output_path, dpi=300)
plt.show()

# ------------------ Distance vs. Threshold Figure ------------------
plt.figure(figsize=(8, 6))
for label, (model_path, method, color) in models.items():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    _, X_test, _, y_test = load_data(method)
    y_test_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_test_score)
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)

    plt.plot(thresholds, distances, lw=1.5, label=f'{label}', color=color)

plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.2)
plt.title('Sn/Sp Distance vs Threshold', fontsize=12)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.xlim([0.2, 0.8])
plt.ylim([0.2, 0.7])
plt.tight_layout()
output_path = os.path.join('mutual_info_classif_XGBoost', f"Threshold.png")
#plt.savefig(output_path, dpi=300)
#plt.show()
