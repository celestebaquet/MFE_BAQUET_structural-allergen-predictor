import matplotlib.pyplot as plt
import numpy as np

def compare_models_radar(metrics_list, labels_list , name, colors_list=None):
    """
    Affiche un radar plot comparant les métriques de plusieurs modèles.

    Args:
        metrics_list (list of list): Chaque sous-liste contient les métriques dans l'ordre suivant :
            [Recall, Spécificité, Accuracy, MCC, F1-Score, BACC, AUC]
        labels_list (list of str): Noms des modèles à afficher dans la légende.
        colors_list (list of str, optional): Couleurs à utiliser. Si None, couleurs par défaut.
        name (str): Nom du set testé
    """
    assert len(metrics_list) == len(labels_list), "Nombre de jeux de métriques et de labels doit correspondre."
    
    num_metrics = 6  
    metric_labels = [
        "Sensitivity",
        "Specificity",
        "MCC",
        "F1-Score",
        "BACC",
        "AUC"
    ]

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    if colors_list is None:
        colors_list = ['tab:blue', 'tab:pink', 'tab:purple']

    for metrics, label, color in zip(metrics_list, labels_list, colors_list):
        data = metrics + [metrics[0]]  # Boucler le radar
        ax.plot(angles, data, label=label, color=color, linewidth=2)
        ax.fill(angles, data, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    # Positionner les labels à l'extérieur du cercle
    for i, label in enumerate(metric_labels):
        angle_rad = angles[i]
        if label == "Sensitivity":
            ax.text(angle_rad, 1.2, label, horizontalalignment='center', verticalalignment='center', fontsize=12)
        elif label == "F1-Score":
            ax.text(angle_rad, 1.15, label, horizontalalignment='center', verticalalignment='center', fontsize=12)
        else:
            ax.text(angle_rad,1.1, label, horizontalalignment='center', verticalalignment='center', fontsize=12)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title(f"Performances of SEP-AlgPro - {name}", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Mettre les résultats dans cet ordre : 
    #["Recall (Sensibilité)", "Spécificité",  "MCC", "F1-Score", "BACC","AUC"]
    results_article_ind4 = [0.958,	0.935,   0.747,	0.760,	0.947,	0.966]
    results_filtre1_ind4 = [0.896,	0.946,   0.611,	0.597,	0.921,	0.949]
    results_filtre2_ind4 = [0.690,	0.952,   0.341,	0.293,	0.821,	0.877]	

    results_article_ind3 = [0.901,	0.949,		0.802,	0.856,	0.925,	0.956]
    results_filtre1_ind3 = [0.786,	0.951,		0.631,	0.653,	0.868,	0.923]
    results_filtre2_ind3 = [0.628,	0.952,		0.434,	0.442,	0.790,	0.865]		

    results_article_ind2 = [0.906,	0.980,   0.844,	0.856,	0.943,	0.960]
    results_filtre1_ind2 = [0.852,	0.980 ,  0.746,	0.754,	0.916,	0.946]
    results_filtre2_ind2 = [0.676,	0.980  , 0.514,	0.510,	0.828,	0.891]	

    results_article_ind1 = [0.938,	0.976,   0.915,	0.956,	0.957,	0.985]
    results_filtre1_ind1 = [0.853,	0.976,   0.853,	0.889,	0.914,	0.946]
    results_filtre2_ind1 = [0.640,	0.976,   0.675,	0.707,	0.808,	0.877]	

    labels_list = ['SEP-AlgPro', 'Filter 1', 'Filter 2']

    name = 'Independant 4'

    compare_models_radar([results_article_ind4, results_filtre1_ind4, results_filtre2_ind4], labels_list, name)
				