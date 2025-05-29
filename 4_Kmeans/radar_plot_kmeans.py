import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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
    
    num_metrics = 5
    metric_labels = [
        "Sensitivity", 
        "Specificity", 
        "MCC", 
        "F1-Score", 
        "BACC"
    ]

    angles = np.linspace(0 + (3/10)*np.pi, 2 * np.pi + 3/10*np.pi , num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    if colors_list is None:
        colors_list = ['steelblue', 'cornflowerblue', 'mediumpurple', 'plum']

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
            ax.text(angle_rad, 1.15, label, horizontalalignment='center', verticalalignment='center', fontsize=12)
        #elif label == "Specificity" or label == 'Accuracy' or label == 'BACC':
            #ax.text(angle_rad, 1.2, label, horizontalalignment='center', verticalalignment='center', fontsize=12)
        else:
            ax.text(angle_rad,1.1, label, horizontalalignment='center', verticalalignment='center', fontsize=12)

    # Affichage plus large : -0.5 à 1
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    ax.set_title(f"Classification of the Training set using K-mean", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), title='Pooling method')

    plt.tight_layout()
    # Juste avant plt.tight_layout() ou plt.show()
    gridlines = ax.yaxis.get_gridlines()
    yticks = ax.get_yticks()

    for ytick, line in zip(yticks, gridlines):
        if ytick == 0 or ytick == 0.5:
            line.set_linewidth(2.5)

    plt.show()

if __name__ == '__main__':
    # Mettre les résultats dans cet ordre : 
    #["Recall (Sensibilité)", "Spécificité", "MCC", "F1-Score", "BACC"]
    results_mean = [0.7739, 0.4535,  	0.2401, 0.6671, 0.6137]
    results_max = [0.6609, 0.7521, 	0.4148, 0.6925,	0.7065]
    results_min = [0.7887, 0.6155, 	0.4104, 0.7259,	0.7021]	
    results_norm = [0.7250,  0.5824,  0.3106,  0.6767, 0.6537]	
    #results_std = [0.3937,  0.4849, -0.1220,  0.4125, 0.4393]
    #results_median = [0.2229, 0.5595, -0.2311, 0.2680, 0.3912]

    labels_list = ['Mean', 'Max', 'Min', 'Norm'] #'Std', 'Median']

    name = 'K-means'

    compare_models_radar([results_mean, results_max, results_min, results_norm], labels_list, name)