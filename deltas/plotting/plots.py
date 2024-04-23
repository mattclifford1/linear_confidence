from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import numpy as np
import deltas.utils.projection as projection
import deltas.utils.radius as radius

# plot colours
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
cm = plt.cm.RdBu
scatter_point_size = 200
font_size = 30
ticks_size = 24
ticks_size_small = 20

def projections_from_data_clfs(clfs, X, y, ax=None):
    ax, show = _get_axes(ax)
    y_plt = 0
    for clf in clfs:
        X_proj = clf.get_projection(X)
        data = {'X': X_proj, 'y': y}

        xp1, xp2 = projection.get_classes(data)
        ax.scatter(xp1, np.ones_like(xp1)*y_plt, c='b',
                s=10, label='Class 1', marker='o')
        ax.scatter(xp2, np.ones_like(xp2)*y_plt, c='r',
                s=10, label='Class 2', marker='x')
        y_plt -= 0.2
        
def plot_projection(data, means=None, R1_emp=None, R2_emp=None, data_info=None, R_est=False, ax=None, deltas_to_plot=[1, 0.5, 0.1, 0.001], calc_data=True):
    ax, _ = _get_axes(ax)

    xp1, xp2 = projection.get_classes(data)
    y = 0
    ax.scatter(xp1, np.ones_like(xp1)*y, c='b', 
               s=10, label=f'Class 1: {xp1.shape[0]}', marker='o')
    ax.scatter(xp2, np.ones_like(xp2)*y, c='r',
               s=10, label=f'Class 2: {xp2.shape[0]}', marker='o')

    # data empircal results
    if calc_data == True:
        # empirical means
        emp_xp1, emp_xp2 = projection.get_emp_means(data)
        ax.scatter(np.array([emp_xp1, emp_xp2]), [y, y], c='k', s=200, 
                marker='x', label='Empircal Means')
        ax.text(emp_xp1, y+.1, r'$\bar{\phi}_{S_1}$', fontsize=12)
        ax.text(emp_xp2, y+.1, r'$\bar{\phi}_{S_2}$', fontsize=12)
        # supports
        # if 'supports' in data.keys():
        #     ax.scatter(data['supports'], [y, y], c='k', s=100,
        #             marker='+', label='Supports')
        
    # expected means
    if means != None:
        ax.scatter(means['X'], [y, y], c='k', s=100,
                marker='o', label='Expected Means')
        
    # empirical estimates of Rs
    if R_est == True:
        name = 'Estimate'
    else:
        name = 'Empirical'
    y = -0.3
    if R_est == False:
        txt1 = r'$\bar{R}_1$'
        txt2 = r'$\bar{R}_2$'
    else:
        txt1 = r'$\hat{\bar{R}}_1$'
        txt2 = r'$\hat{\bar{R}}_2$'
    if R1_emp != None:
        ax.plot([emp_xp1, emp_xp1+R1_emp], [y, y],
                c='b', label=f'R1 {name}', marker='|')
        ax.text(emp_xp1+(R1_emp/2), y+.05, txt1, fontsize=12)
    if R2_emp != None:
        ax.plot([emp_xp2-R2_emp, emp_xp2], [y, y], 
                c='r', label=f'R2 {name}', marker='|')
        ax.text(emp_xp2-(R2_emp/2), y+.05, txt2, fontsize=12)
    
    # plot R error estimates if given the data
    deltas_to_plot = [0.9999999999999999999]
    y = 0.4
    if data_info != None:
        for d in deltas_to_plot:
            R_ests = get_R_estimates(data_info, deltas=[d, d])
            ax.plot([emp_xp1, emp_xp1+R_ests[0]], [y, y],
                    c='b', marker='|', linestyle='dashed')
            ax.text(emp_xp1+((emp_xp1+R_ests[0])/2),
                    y+.05, r'$\hat{\bar{R}}_1, \delta_1=0.\dot{9}$', fontsize=10)
            ax.plot([emp_xp2-R_ests[1], emp_xp2], [y, y],
                    c='r', marker='|', linestyle='dashed')
            ax.text(emp_xp2-((emp_xp2+R_ests[1])/2),
                    y+.05, r'$\hat{\bar{R}}_2, \delta_2=0.\dot{9}$', fontsize=10)
        ax.set_ylabel('deltas (dashed)')
            
    # to make plot scale nice with a legend
    ax.plot([0], [2], c='w')
    # ax.legend(loc='upper right')

    ax.set_xlabel(r'$< \phi(x), w >$')
    ax.set_ylabel('')
    ax.set_yticks([])
    # ax.autoscale(enable=True, axis='y', tight=True)
    return ax

def get_R_estimates(data_info, deltas=[1, 1]):
    R1_est = radius.R_upper_bound(data_info['empirical R1'], data_info['R all data'], data_info['N1'], deltas[0])
    R2_est = radius.R_upper_bound(data_info['empirical R2'], data_info['R all data'], data_info['N2'], deltas[1])
    return R1_est, R2_est

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
        x_txt = 'Feature 1'
        y_txt = 'Feature 2'
    else:
        X = dim_reducer.transform(data['X'])
        x1 = list(X[:, 0])
        x2 = list(X[:, 1])
        x_txt = 'PCA 1'
        y_txt = 'PCA 2'
    # count the occurrences of each point
    point_count = Counter(zip(x1, x2))
    # create a list of the sizes, here multiplied by 10 for scale
    size = [scatter_point_size *
            point_count[(xx1, xx2)] for xx1, xx2 in zip(x1, x2)]

    ax.scatter(x1, x2, s=scatter_point_size,
               c=data['y'], cmap=cm_bright, edgecolors="k", alpha=0.4,
               linewidths=2)
    ax.grid(False)
    ax.set_xlabel(x_txt, size=ticks_size)
    ax.set_ylabel(y_txt, size=ticks_size)
    if show == True:
        plt.show()


def plot_decision_boundary_custom_pred(pred_func, data, ax=None, dim_reducer=None, labels=True, probs=True):
    ''' for pred given delta1 func'''
    xgrids, zz = get_grid_pred(
        None, data, probs=probs, dim_reducer=dim_reducer, custom_pred=pred_func)
    _plot_decision_boundary(xgrids, zz, ax=None, labels=True, probs=probs)


def plot_decision_boundary(clf, data, ax=None, dim_reducer=None, labels=True, probs=True):
    '''
    plot a decision boundary on axes
    input:
        - clf: sklearn classifier object
    '''
    xgrids, zz = get_grid_pred(clf, data, probs=probs, dim_reducer=dim_reducer)
    _plot_decision_boundary(xgrids, zz, ax=None, labels=True, probs=probs)


def _plot_decision_boundary(xgrids, zz, ax=None, labels=True, probs=True):
    
    # if dim_reducer != None:
    #     X = dim_reducer.transform(data['X'])
    #     # need to impliment this to encorportant n dimensions
    #     print('non impliment')
    #     return

    ax, show = _get_axes(ax)
    
    # plot the grid of x, y and z values as a surface
    c = ax.contourf(xgrids[0], xgrids[1], zz, cmap=cm,
                    vmin=0, vmax=1, alpha=0.7)
    c.set_clim(0, 1)

    # add a legend, called a color bar
    if probs == True:
        cbar_label = 'Probability'
        ticks = [0, 0.5, 1]
    else:
        cbar_label = 'Predicted Class'
        ticks = [0, 1]
    cbar = plt.colorbar(c, ticks=ticks)
    if labels == True:
        cbar.ax.tick_params(labelsize=ticks_size)
        cbar.ax.set_ylabel(cbar_label, size=ticks_size)

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

def get_grid_pred(clf, data, probs=True, dim_reducer=None, flat=False, res=25, custom_pred=None):
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
    if custom_pred != None:
        # do opposite to match probs colours where we show P(X=class 0)
        yhat = 1 - custom_pred(flat_grid)
    elif probs == True:
        yhat = clf.predict_proba(flat_grid)
        # keep just the probabilities for class 0
        yhat = yhat[:, 0]
    else:
        # do opposite to match probs colours where we show P(X=class 0)
        yhat = 1 - clf.predict(flat_grid)

    # yhat = clf.predict(flat_grid)

    # reshape the predictions back into a grid
    if flat == False:
        zz = yhat.reshape(xgrids[0].shape)
        return xgrids, zz
    else:
        return flat_grid, yhat


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

def deltas_projected_boundary(delta1, delta2, data_info):
    # calculate each R upper bound
    R1_est = radius.R_upper_bound(
        data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
    R2_est = radius.R_upper_bound(
        data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta2)
    _, ax = plt.subplots(1, 1)
    _ = plot_projection(
        data_info['projected_data'], None, R1_est, R2_est, R_est=True, ax=ax)
