import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import numpy as np


def hexplot_2features(x_idx, y_idx, model_fname_list, m_f_mat, mat_mask):
    '''
    Function to plot 2D histogram of 2 variables and their respective 1D histogram with seaborn
    ToDo: Need to position R2 value better (try to  move it left)
    '''

    import seaborn as sns

    x = m_f_mat[:, x_idx][mat_mask]
    y = m_f_mat[:, y_idx][mat_mask]

    hexplot = sns.jointplot(
        x=x,
        y=y,
        kind="hex",
        bins="log",
        mincnt=1,
        marginal_kws=dict(
            bins=50,
            color="black"),
        joint_kws={
            "color": None,
            "cmap": 'viridis'})
    hexplot.set_axis_labels(
        model_fname_list[x_idx], model_fname_list[y_idx], fontsize=14)

    # shrink fig so cbar will be visible
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Calculate R2
    correlation_xy = np.corrcoef(x, y)[0, 1]
    r_squared = correlation_xy ** 2
    r_squared_rounded = round(r_squared, 2)

    # Put R2 on plot in a box
    plt.text(
        0.2,
        y.max(),
        "R2= {}".format(r_squared_rounded),
        fontsize=10,
        horizontalalignment='center',
        verticalalignment='center',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round'))

    # make new ax object for the cbar and add colorbar to it
    # x, y, width, height (adds an axes where the  colorbar will be drawn)
    cbar_ax = hexplot.fig.add_axes([.85, 0.2, .025, .5])
    # cax specifies the axes onto which colorbar will be drawn
    plt.colorbar(cax=cbar_ax)

    plt.show()


def pairplots_allfeatures(model_fname_list, m_f_mat, mat_mask):
    ''''
    Function to make pairplots of all features (with kde plots on diagonal)
    '''

    import seaborn as sns
    # Convert numpy array to pandas dataframe to be used by seaborn pairplot
    np_array = m_f_mat[mat_mask]
    df = pd.DataFrame(np_array, columns=model_fname_list)

    # Make plots
    sns.pairplot(
        df,
        kind='hist',
        diag_kind='kde',
        corner=True,
        diag_kws=dict(
            fill=False),
        plot_kws=dict(
            color=None,
            cmap="viridis",
            cbar=True,
            norm=LogNorm(),
            vmin=None,
            vmax=None))
