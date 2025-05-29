import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
import pickle
import glob

def select_features(X, selected_features):
    return X[:, selected_features]

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

def get_results(concat_methods, selector, classifier, threshold=0.3):
    metrics_all = []
    concat_labels = []

    for method in concat_methods:
        model_path = f'{selector}_{classifier}/best_model_{selector}_XGBClassifier_{method}_k*.pkl'
        # If wildcard used, find exact match

        matched = glob.glob(model_path)
        if not matched:
            print(f"Model not found for {method}")
            continue
        model_path = matched[0]

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load test data
        X_test = np.load(f'Ind1_F2/stacked_{method}.npy')
        y_test = np.load(f'Ind1_F2/labels.npy')

        # Predict with threshold
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Metrics
        sn = tp / (tp + fn) if tp + fn else 0
        sp = tn / (tn + fp) if tn + fp else 0
        bacc = (sn + sp) / 2
        dist = np.sqrt((1 - sn)**2 + (1 - sp)**2)

        metrics_all.append([sn, sp, bacc, dist])
        concat_labels.append(method)

    # Convert to numpy arrays
    metrics_all = np.array(metrics_all).T  # Transpose to get metrics per row
    return metrics_all, concat_labels


def main(concat_method, selector, classifier, metrics_show):
    repository_name = f'{selector}_{classifier}'
    os.makedirs(repository_name, exist_ok=True)

    best_results, concat_labels = get_results(concat_method, selector, classifier, threshold=0.33)
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
    annotate_heatmap(im1, data=metrics_data, valfmt="{x:.3f}", fontsize=16)

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
    annotate_heatmap(im2, data=distance_data, valfmt="{x:.3f}", fontsize=16, textcolors=("white", "black"))

    # Tighter title
    fig.suptitle(f"Heatmap of Model Performances ({selector}/{classifier}) with a 0.33 threshold", fontsize=16)

    output_path = os.path.join(repository_name, f"heatmap_{selector}_{classifier}_0_33.png")
    #plt.savefig(output_path, dpi=300)
    plt.show()



if __name__ == '__main__':
    concat_method = ['mean', 'max', 'min', 'median', 'norm', 'std']
    metrics_show = ['Sn', 'Sp', 'BACC', 'Sn/Sp dist.']
    selector = 'RFE'
    classifier = 'XGBoost'
    main(concat_method, selector, classifier, metrics_show)
