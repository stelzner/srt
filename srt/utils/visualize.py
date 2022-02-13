import numpy as np
from matplotlib import pyplot as plt

from colorsys import hsv_to_rgb


def background_image(shape, gridsize=2, lg=0.85, dg=0.5):
    bg = np.zeros(shape)
    c1 = np.array((lg, lg, lg))
    c2 = np.array((dg, dg, dg))

    for i, x in enumerate(range(0, shape[0], gridsize)):
        for j, y in enumerate(range(0, shape[1], gridsize)):
            c = c1 if (i + j) % 2 == 0 else c2
            bg[x:x+gridsize, y:y+gridsize] = c

    return bg


def visualize_2d_cluster(clustering, colors=None):
    if colors is None:
        num_clusters = clustering.max()
        colors = get_clustering_colors(num_clusters)
    img = colors[clustering]
    return img


def get_clustering_colors(num_colors):
    colors = [(0., 0., 0.)]
    for i in range(num_colors):
        colors.append(hsv_to_rgb(i / num_colors, 0.45, 0.8))
    colors = np.array(colors)
    return colors


def setup_axis(axis):
    axis.tick_params(axis='both',       # changes apply to the x-axis
                     which='both',      # both major and minor ticks are affected
                     bottom=False,      # ticks along the bottom edge are off
                     top=False,         # ticks along the top edge are off
                     right=False,
                     left=False,
                     labelbottom=False,
                     labelleft=False)   # labels along the bottom edge are off


def draw_visualization_grid(columns, outfile, row_labels=None, name=None):
    num_rows = columns[0][1].shape[0]
    num_cols = len(columns)
    num_segments = 1

    bg_image = None
    imshow_args = {'interpolation': 'none', 'cmap': 'gray'}

    for i in range(num_cols):
        column_type = columns[i][2]
        if column_type == 'clustering':
            num_segments = max(num_segments, columns[i][1].max())
        if column_type == 'image' and bg_image is None:
            bg_image = background_image(list(columns[i][1].shape[1:3]) + [3])

    colors = get_clustering_colors(num_segments)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for c in range(num_cols):
        axes[0, c].set_title(columns[c][0])
        col_type = columns[c][2]
        for r in range(num_rows):
            setup_axis(axes[r, c])
            img = columns[c][1][r]
            if col_type == 'image':
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                axes[r, c].imshow(bg_image, **imshow_args)
                axes[r, c].imshow(img, **imshow_args)
                if len(columns[c]) > 3:
                    axes[r, c].set_xlabel(columns[c][3][r])
            elif col_type == 'clustering':
                axes[r, c].imshow(visualize_2d_cluster(img, colors), **imshow_args)

    if row_labels is not None:
        for r in range(num_rows):
            axes[r, 0].set_ylabel(row_labels[r])

    plt.savefig(f'{outfile}.png')
    plt.close()


