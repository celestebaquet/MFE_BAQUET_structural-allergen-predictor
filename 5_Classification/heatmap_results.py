import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    if ax is None:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(range(data.shape[1]), labels=col_labels)
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

def get_best_result_from_csv(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Warning] Fichier introuvable : {csv_path}")
        return None

    distances = ((1 - df['specificity'])**2 + (1 - df['sensitivity'])**2)**0.5
    best_idx = distances.idxmin()
    best_row = df.loc[best_idx].to_dict()
    return best_row

def get_results(concat_method, selector, classifier):
    best_results = []
    concat_labels = concat_method.copy()  # pour ne pas modifier l'original
    concat_labels = [label.capitalize() for label in concat_labels]


    for concat in concat_method:
        csv_path = f'{selector}_{classifier}/results_{selector}_{classifier}_{concat}.csv'
        result = get_best_result_from_csv(csv_path)
        if result is None:
            continue

        sn_sp_dist = ((1 - result['specificity'])**2 + (1 - result['sensitivity'])**2)**0.5

        best_results.append([
            result['sensitivity'],
            result['specificity'],
            result['bacc'],
            sn_sp_dist
        ])

    # Ajout de la méthode de référence (SEP)
    #SEP_sn = 0.640
    #SEP_sp = 0.976
    #SEP_bacc = 0.808
    #sn_sp_dist_sep = ((1 - SEP_sp)**2 + (1 - SEP_sn)**2)**0.5

    #best_results.append([
    #    SEP_sn,
    #    SEP_sp,
    #    SEP_bacc,
    #    sn_sp_dist_sep
    #])
    #concat_labels.append('SEP-AlgPro')

    data = np.array(best_results).T
    return data, concat_labels

def main(concat_method, selector, classifier, metrics_show):
    repository_name = f'{selector}_{classifier}'
    os.makedirs(repository_name, exist_ok=True)

    best_results, concat_labels = get_results(concat_method, selector, classifier)
    metrics_data = best_results[:3]
    distance_data = best_results[3:]
    concat_labels = [label.capitalize() for label in concat_labels]

    fig = plt.figure(figsize=(10, 5.5), constrained_layout=True)  # Smaller height + auto layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

    # First Heatmap
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(metrics_data, cmap="BuPu", aspect="equal", vmin=0.6)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.02)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label("Score", fontsize=14)

    ax1.set_xticks(range(len(concat_labels)), labels=concat_labels)
    ax1.set_yticks(range(3), labels=metrics_show[:3])
    ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax1.set_xticks(np.arange(metrics_data.shape[1] + 1) - .5, minor=True)
    ax1.set_yticks(np.arange(metrics_data.shape[0] + 1) - .5, minor=True)
    ax1.set_xticklabels(concat_labels, fontsize=16)
    ax1.set_yticklabels(metrics_show[:3], fontsize=16)
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax1.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(im1, data=metrics_data, valfmt="{x:.2f}", fontsize=16)

    # Second Heatmap
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(distance_data, cmap="Purples_r", aspect="equal", vmax=0.4)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.02)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label("Distance", fontsize=14)

    ax2.set_xticks([])
    ax2.set_yticks([0], labels=[metrics_show[3]])
    ax2.set_yticklabels([metrics_show[3]], fontsize=16)
    ax2.set_xticks(np.arange(distance_data.shape[1] + 1) - .5, minor=True)
    ax2.set_yticks(np.arange(distance_data.shape[0] + 1) - .5, minor=True)
    ax2.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax2.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(im2, data=distance_data, valfmt="{x:.2f}", fontsize=16, textcolors=("white", "black"))

    # Tighter title
    fig.suptitle(f"Heatmap of Model Performances ({selector}/{classifier})", fontsize=16)

    output_path = os.path.join(repository_name, f"heatmap_{selector}_{classifier}.png")
    plt.savefig(output_path, dpi=300)
    #plt.show()



if __name__ == '__main__':
    concat_method = ['mean', 'max', 'min', 'median', 'norm', 'std']
    metrics_show = ['Sn', 'Sp', 'BACC', 'Sn/Sp dist.']
    selector = 'f_classif'
    classifier = 'GB_default'
    main(concat_method, selector, classifier, metrics_show)
