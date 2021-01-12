import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

datasets = ["OBESITY", "GSE75473", "GSE112804", "GSE126386", "GSE129373"]
classifiers = ["DT", "LR", "RF", "NB", "NN", "GBT", "SVM"]


accuracy_default = np.array([[45.455, 40.909, 54.545, 40.909, 45.455, 31.818, 63.636],
                    [57.134, 45.238, 59.524, 50, 52.381, 64.286, 50],
                    [70, 83.333, 76.667, 83.333, 86.667, 80, 86.667],
                    [66.667, 60, 66.667, 56.667, 70, 73.333, 53.333],
                    [77.778, 81.481, 74.074, 92.593, 77.778, 49.69, 77.778]])


accuracy_finetuned = np.array([[54.545, 40.909, 59.091, 54.545, 63.636, 36.364, 63.636],
                    [57.134, 47.619, 59.524, 50, 52.381, 64.286, 57.143],
                    [80, 90, 83.333, 86.667, 90, 80, 86.667],
                    [66.667, 66.667, 66.667, 63.333, 70, 73.333, 60],
                    [77.778, 88.889, 74.074, 92.593, 85.185, 62.963, 81.481]])

#fig, ax = plt.subplots()
#im = ax.imshow(accuracy_finetuned)

#ax.set_xticks(np.arange(len(classifiers)))
#ax.set_yticks(np.arange(len(datasets)))
#ax.set_xticklabels(classifiers)
#ax.set_yticklabels(datasets)

#for i in range(len(datasets)):
#    for j in range(len(classifiers)):
#        text = ax.text(j, i, accuracy_finetuned[i, j],
#                       ha="center", va="center", color="w")

#ax.set_title("Harvest of local farmers (in tons/year)")
#fig.tight_layout()
#plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()

im, cbar = heatmap(accuracy_finetuned, datasets, classifiers, ax=ax, cmap="YlGn", cbarlabel="accuracy")

texts = annotate_heatmap(im, valfmt="{x:.3f}")

fig.tight_layout()
plt.show()