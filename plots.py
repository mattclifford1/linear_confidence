from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import numpy as np

# plot colours
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
cm = plt.cm.RdBu
scatter_point_size = 200
font_size = 30
ticks_size = 24
ticks_size_small = 20


def plot_projection(data, means, R1_emp, R2_emp,  ax=None):
    ax, show = _get_axes(ax)

    # ax.scatter(decision_projected, np.zeros_like(
    #     decision_projected), c='k', s=50, marker='x')

    # data points
    xp1 = np.array([p for i, p in enumerate(data['X']) if data['y'][i] == 0])
    xp2 = np.array([p for i, p in enumerate(data['X']) if data['y'][i] == 1])
    ax.scatter(xp1,np.zeros_like(xp1),c='b',s=10, label='Class 1')
    ax.scatter(xp2, np.zeros_like(xp2), c='r', s=10, label='Class 2')
    # empirical means
    emp_xp1 = np.mean(xp1)
    emp_xp2 = np.mean(xp2)
    ax.scatter(np.array([emp_xp1, emp_xp2]), [0, 0], c='k', s=200, 
               marker='x', label='Empircal Means')
    # expected means
    ax.scatter(means['X'], [0, 0], c='k', s=100,
               marker='o', label='Expected Means')
    # supports
    ax.scatter(data['supports'], [0, 0], c='k', s=100,
               marker='+', label='Supports')
    
    # empirical estimates of Rs
    ax.plot([means['X'][0]-R1_emp, means['X'][0]+R1_emp], [-0.1, -0.1], c='b', label='R1 empirical', marker='|')
    ax.plot([means['X'][1]-R1_emp, means['X'][1]+R2_emp], [-0.1, -0.1], c='r', label='R2 empirical', marker='|')

    ax.scatter([0], [-1], c='w')
    ax.legend()
    return ax


def plot_classes(data, ax=None, dim_reducer=None):
    '''
    plot classes in different colour on an axes, duplicate points in the data are enlarged for clarity
    input:
        - data: dictionary with keys 'X', 'y'
    '''
    ax, show = _get_axes(ax)
    if dim_reducer == None:
        x1 = list(data['X'][:, 0])
        x2 = list(data['X'][:, 1])
    else:
        X = dim_reducer.transform(data['X'])
        x1 = list(X[:, 0])
        x2 = list(X[:, 1])
    # count the occurrences of each point
    point_count = Counter(zip(x1, x2))
    # create a list of the sizes, here multiplied by 10 for scale
    size = [scatter_point_size *
            point_count[(xx1, xx2)] for xx1, xx2 in zip(x1, x2)]

    ax.scatter(x1, x2, s=scatter_point_size,
               c=data['y'], cmap=cm_bright, edgecolors="k", alpha=0.4,
               linewidths=2)
    ax.grid(False)
    if show == True:
        plt.show()


def plot_decision_boundary(clf, data, ax=None, dim_reducer=None, labels=True):
    '''
    plot a decision boundary on axes
    input:
        - clf: sklearn classifier object
    '''
    # if dim_reducer != None:
    #     X = dim_reducer.transform(data['X'])
    #     # need to impliment this to encorportant n dimensions
    #     print('non impliment')
    #     return

    ax, show = _get_axes(ax)
    # get X from data
    X = data['X']
    if dim_reducer != None:
        X = dim_reducer.transform(X)
    n_features = X.shape[1]

    # define bounds of the domain
    lims = []
    for f in range(n_features):
        min = X[:, f].min()-1
        max = X[:, f].max()+1
        lims.append((min, max))

    # define the x and y scale
    res = 25
    xranges = []
    for f in range(n_features):
        step = (lims[f][1] - lims[f][0]) / res
        xranges.append(np.arange(lims[f][0], lims[f][1], step))

    # create all of the lines and rows of the grid
    xgrids = np.meshgrid(*xranges)
    # flatten each grid to a vector
    flats = []
    for f in range(n_features):
        flats.append(xgrids[f].flatten().reshape(-1, 1))

    # horizontal stack vectors to create x1, x2 input for the model
    flat_grid = np.hstack(flats)
    if dim_reducer != None:
        flat_grid = dim_reducer.inverse_transform(flat_grid)

    # make predictions for the flat_grid
    yhat = clf.predict_proba(flat_grid)
    # keep just the probabilities for class 0
    yhat = yhat[:, 0]

    # yhat = clf.predict(flat_grid)

    # reshape the predictions back into a grid
    zz = yhat.reshape(xgrids[0].shape)

    # plot the grid of x, y and z values as a surface
    c = ax.contourf(xgrids[0], xgrids[1], zz, cmap=cm,
                    vmin=0, vmax=1, alpha=0.7)
    c.set_clim(0, 1)

    # add a legend, called a color bar
    cbar = plt.colorbar(c, ticks=[0, 0.5, 1])
    if labels == True:
        cbar.ax.tick_params(labelsize=ticks_size)
        cbar.ax.set_ylabel('Probability', size=ticks_size)

    # set labels
    # if labels == True:
    #     if dim_reducer != None:
    #         ax.set_xlabel('PCA Component 1', fontsize=font_size)
    #         ax.set_ylabel('PCA Component 2', fontsize=font_size)
    #     else:
    #         ax.set_xlabel('Feature 1', fontsize=font_size)
    #         ax.set_ylabel('Feature 2', fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=ticks_size_small)

    if show == True:
        plt.show()


def _get_axes(ax):
    '''
    determine whether to make an axes or not, making axes also means show them
    input:
        - ax: None or matplotlib axes object
    '''
    if ax == None:
        ax = plt.gca()
        show = True
    else:
        show = False
    return ax, show
