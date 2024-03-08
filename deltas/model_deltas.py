'''
scikit-learn style class to fit deltas
'''
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

import deltas.data.utils as utils
import deltas.plotting.plots as plots
import deltas.data.normal as normal
import deltas.classifiers.models as models
import deltas.utils.projection as projection
import deltas.utils.radius as radius
import deltas.utils.equations as ds
import deltas.optimisation.optimise_contraint as optimise_contraint
import deltas.optimisation.optimise_deltas as optimise_deltas


class base_deltas:
    def __init__(self, clf):
        if not hasattr(clf, 'get_projection'):
            raise AttributeError(f"Classifier {clf} needs 'get_projection' method")
        self.clf = clf
        self._setup()

    def _setup(self):
        self.data_info_made = False
        self.is_fit = False
        # deltas optimisation functions
        self.loss_func = ds.loss_one_delta
        self.contraint_func = ds.contraint_eq7
        self.delta2_from_delta1 = ds.delta2_given_delta1_matt
        self.delta1_from_delta2 = ds.delta1_given_delta2_matt

    def fit(self, X, y, costs=(1, 1), _plot=False, _print=False):
        # Make data_info - R_ests, D, etc.
        self.data_info = self.get_data_info(X, y, self.clf, costs, _print=_print)
        self.data_info_made = True

        # optimise for the deltas
        res = self._optimise(self.data_info, 
                             self.loss_func,
                             self.contraint_func,
                             self.delta2_from_delta1,
                             _plot=_plot, 
                             _print=_print)
        self.delta1 = res['delta1']
        self.delta2 = res['delta2']
        self.solution_possible = res['solution_possible']
        self.solution_found = res['solution_found']

        # make boundary 
        self.boundary, self.class_nums = self._make_boundary(
            self.delta1, self.delta2)
        self.is_fit = True

        return self
    
    def _make_boundary(self, delta1, delta2):
        # calculate each R upper bound
        R1_est = radius.R_upper_bound(
            self.data_info['empirical R1'], self.data_info['R all data'], self.data_info['N1'], delta1)
        R2_est = radius.R_upper_bound(
            self.data_info['empirical R2'], self.data_info['R all data'], self.data_info['N2'], delta2)
        Rs = {'R1': R1_est, 'R2': R2_est}

        # add error to min class and minus from max class
        means = [self.data_info['empirical_projected_mean 1'], self.data_info['empirical_projected_mean 2']]
        min_mean = np.argmin(means)
        max_mean = np.argmax(means)
        class_names = ['1', '2']
        upper_min_class = self.data_info[f'empirical_projected_mean {class_names[min_mean]}'] + \
            Rs[f'R{class_names[min_mean]}']
        lower_max_class = self.data_info[f'empirical_projected_mean {class_names[max_mean]}'] - \
            Rs[f'R{class_names[max_mean]}']
    
        # get average as the boundary
        boundary = (upper_min_class + lower_max_class)/2
        class_nums = [0, 1]
        class_nums = [class_nums[min_mean], 
                      class_nums[max_mean]]
        return boundary, class_nums
    
    def predict(self, X):
        if self.is_fit == False:
            raise AttributeError("Not fit to any data yet, call 'fit(X, y)' or  method first")
        return self._predict(X, self.boundary, self.class_nums)
        
    def _predict(self, X, boundary, class_nums):
        # project data if not already
        if X.shape[1] != 1:
            if self.clf != None:
                X = self.clf.get_projection(X)
            else:
                raise AttributeError(
                    f"Deltas classifier needs original classifier to project feature space onto 1D classification space")
        preds = np.zeros(X.shape)
        preds[X <= boundary] = class_nums[0]
        preds[X > boundary] = class_nums[1]
        return preds.squeeze()

    def _predict_given_delta1(self, X, delta1):
        delta2 = self.delta2_from_delta1(delta1, self.data_info)
        boundary, class_nums = self._make_boundary(delta1, delta2)
        return self._predict(X, boundary, class_nums)
    
    def _predict_given_delta2(self, X, delta2):
        delta1 = self.delta1_from_delta2(delta2, self.data_info)
        boundary, class_nums = self._make_boundary(delta1, delta2)
        return self._predict(X, boundary, class_nums)

    @staticmethod
    def get_data_info(X, y, clf, costs=(1, 1), _print=False):
        # project data according to classifier and calculate data attributes needed
        data = {'X': X, 'y': y}
        proj_data = projection.from_clf(data, clf, supports=True)
        # Empircal M
        M_emp = np.abs(proj_data['supports'][1]-proj_data['supports'][0]).squeeze()

        # get Rs
        # R_sup = radius.supremum(data['X'])
        R_sup = radius.supremum(proj_data['X'])
        # empirical means
        xp1, xp2 = projection.get_classes(proj_data)
        emp_xp1, emp_xp2 = projection.get_emp_means(proj_data)
        R1_emp = radius.supremum(proj_data['X1'], emp_xp1)
        R2_emp = radius.supremum(proj_data['X2'], emp_xp2)

        # Empirical D
        D_emp = np.abs(emp_xp1 - emp_xp2)

        data_info = {'projected_data': proj_data,
                     'empirical margin': M_emp,
                     'R all data': R_sup,
                     'projected_data 1': xp1,
                     'projected_data 2': xp2,
                     'empirical_projected_mean 1': emp_xp1,
                     'empirical_projected_mean 2': emp_xp2,
                     'empirical R1': R1_emp,
                     'empirical R2': R2_emp,
                     'empirical D': D_emp,
                     'N1': (data['y'] == 0).sum(),
                     'N2': (data['y'] == 1).sum(),
                     'c1': costs[0],
                     'c2': costs[1],
                    }
        # if _print == True:
        #     print(f'R1 empirical: {R1_emp}\nR2 empirical: {R2_emp}')
        return data_info
    
    def _optimise(self, 
                  data_info, 
                  loss_func, 
                  contraint_func,
                  delta2_from_delta1=None,
                  num_deltas=1, 
                  grid_search=True,
                  _plot=False, 
                  _print=False):
        # optimise for the deltas. N.B. keep data_info as arg for flexibility
        res = optimise_deltas.optimise(
            data_info, 
            loss_func, 
            contraint_func, 
            delta2_from_delta1, 
            num_deltas, 
            grid_search, 
            _print, 
            _plot)

        return res


    def plot_data(self, data_clf=None, m1=None, m2=None):
        if self.data_info_made == True:
            self._plot_data(self.data_info, self.clf, data_clf=data_clf, m1=m1, m2=m2)
        else:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")

    @staticmethod
    def _plot_data(data_info, clf, data_clf=None, m1=None, m2=None):
        # project means if we have them
            proj_means = None
            if isinstance(data_clf, dict):
                if 'mean1' in data_clf.keys() and 'mean2' in data_clf.keys():
                    m1 = data_clf['mean1']
                    m2 = data_clf['mean2']
                    proj_means = projection.from_clf({'X': np.array([m1, m2]), 'y': [0, 1]}, clf)
            # plot the data
            _ = plots.plot_projection(
                data_info['projected_data'], 
                proj_means,
                data_info['empirical R1'],
                data_info['empirical R1'],
                deltas_to_plot=[1],
                data_info=data_info)
            plots.plt.show()


    def print_params(self):
        if self.data_info_made == True:
            print(
                f"""Parameters
                R:  {self.data_info['R all data']}
                N1: {self.data_info['N1']}
                N2: {self.data_info['N2']}
                R1: {self.data_info['empirical R1']}
                R2: {self.data_info['empirical R2']}
                M:  {self.data_info['empirical margin']}
                D:  {self.data_info['empirical D']}
                C1: {self.data_info['c1']}
                C2: {self.data_info['c2']}""")
        else:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")


class reprojectioner:
    '''
    get a project using a model (eg. SVM)
    '''

    def __init__(self, projecter=models.SVM(kernel='rbf')):
        # projecter need to have .fit() and .get_projection attributes
        if not hasattr(projecter, 'get_projection'):
            raise AttributeError(
                f"Projection Model {projecter} needs 'get_projection' method")
        if not hasattr(projecter, 'fit'):
            raise AttributeError(
                f"Projection Model {projecter} needs 'fit' method")
        self.projecter = projecter

    def fit_projection(self, clf, X, y):
        self.clf_original = clf
        X_proj = self.clf_original.get_projection(X)
        # fit reprojection model to the projected data
        self.projecter = self.projecter.fit(X_proj, y)
        return self

    def get_projection(self, X):
        # project with original model
        X_orig_clf = self.clf_original.get_projection(X)
        # reproject with second model (hopefully separated/bigger margin)
        return self.projecter.get_projection(X_orig_clf)
    

class reprojection_deltas(base_deltas):
    '''
    use a model (e.g SVM) to project current classisfier onto 1D projection
    This method doesn't have the requirement that the classifier needs a get_projection method to 1D,
    but it does need the classifier to project to a feature space (not implimented yet)
    '''

    def __init__(self, clf, projection_model=reprojectioner, *args, **kwargs):
        super().__init__(clf, *args, **kwargs)
        self.projection_model = projection_model

    def fit(self, X, y, _plot=False, **kwargs):
        clf_copy = self.clf
        self.clf = self.projection_model.fit_projection(clf_copy, X, y)

        if _plot == True:
            plots.projections_from_data_clfs([clf_copy, self.clf], X, y)

        super().fit(X, y, _plot=_plot, **kwargs)
        return self


class downsample_deltas(base_deltas):
    '''
    Downsample the dataset (randomly) until we find a good solution
    '''

    def __init__(self, clf, *args, **kwargs):
        super().__init__(clf, *args, **kwargs)

    def check_if_solvable(self, data_info):
        '''check the constraint to see if we have a viable solution'''
        if self.contraint_func(1, 1, data_info) <= 0:
            return True
        else:
            return False
        
    def random_downsample_data(self, X, y):
        '''randomly downsample the dataset'''
        # split into each class
        x1 = X[y==0, :]
        x2 = X[y==1, :]
        y1 = y[y==0]
        y2 = y[y==1]
        # downsample each class
        _x1, _y1, num_reduced1 = self.random_downsample_class(x1, y1)
        _x2, _y2, num_reduced2 = self.random_downsample_class(x2, y2)
        # put data back together - don't need to shuffle as deltas algorithm isn't affected by this
        return np.concatenate([_x1, _x2], axis=0), np.concatenate([_y1, _y2], axis=0), num_reduced1, num_reduced2

    def random_downsample_class(self, X, y):
        ' given only one class'
        num_samples = len(y)
        num_down_to = random.randint(1, num_samples)
        num_reduced = num_samples - num_down_to
        # now downsample
        _X, _y = resample(X, y, n_samples=num_down_to)
        return _X, _y, num_reduced


    def _fit(self, data_info, _plot=False, _print=False):
        '''fit so downsampled dataset - dont save to self as we might not use this fit trial'''
        # optimise for the deltas
        res = self._optimise(data_info,
                             self.loss_func,
                             self.contraint_func,
                             self.delta2_from_delta1,
                             _plot=_plot,
                             _print=_print)
        return res


    def fit(self, X, y, costs=(1, 1), alpha=1, cut_off_trials=100, max_trials=10000, force_downsample=False, _plot=False, _print=False):
        '''
        fit to downsampled datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            cut_off_trials:   the number of viable downsampled datasets to find before stopping
            max_trials:       the number of downsampled datasets to try 
            force_downsample: try downsampling even if the original projection is solvable
        '''
        self._fit_single_thread(X, y, costs=costs, alpha=alpha,
                                cut_off_trials=cut_off_trials, max_trials=max_trials, force_downsample=force_downsample,
                                _plot=_plot, _print=_print)
        if _plot == True and self.is_fit == True:
            plots.deltas_projected_boundary(self.delta1, self.delta2, self.data_info)
        return self
        
    def _fit_single_thread(self, X, y, costs, alpha, cut_off_trials, max_trials, force_downsample, _plot=False, _print=True):
        ''' use one worker in simple loop to find solution of random downsampling'''
        best_loss = None
        best_num_points_removed = None
        # check we don't already have solvable without downsampling
        data_info = self.get_data_info(X, y, self.clf, costs, _print=False)
        data_info['num_reduced'] = 0
        best_loss, best_num_points_removed, original_solvable = self._check_and_optimise_data(
            data_info, best_loss, best_num_points_removed)
        if _plot == True:
            print('Original Data')
            self._plot_data(data_info, self.clf)

        # now try as many random downsamples of the dataset as the budget allows
        found_count = 0
        if original_solvable == False or force_downsample == True:
            for _ in tqdm(range(max_trials), desc='Trying random downsampling deltas', leave=False):
                # downsample
                _X, _y, num_reduced1, num_reduced2 = self.random_downsample_data(
                    X, y)
                # see if we can fit deltas
                data_info = self.get_data_info(_X, _y, self.clf, costs, _print=False)
                data_info['num_reduced'] = num_reduced1 + num_reduced2
                data_info['alpha'] = alpha
                best_loss, best_num_points_removed, found = self._check_and_optimise_data(
                    data_info, best_loss, best_num_points_removed)
                # see if we have found enough viable solutions
                if found == True:
                    found_count += 1
                if found_count >= cut_off_trials:
                    break
        else:
            if _print == True:
                print(
                    "Original dataset is solvable so not downsampling, set 'force_downsample' to 'True' to try and find a lower loss via downsampling anyway")
            

        # finished search, now make new boundary if we found a solution
        if best_loss == None:
            if _print == True:
                print('Unable to find result with downsample, increase the budget')
            self.is_fit = False
        else:
            if _plot == True and found_count > 0:
                print(
                    f'Best Random Downsampled dataset solution found with budget: {max_trials} and {found_count} found viable downsampled solution')
                # self._plot_data(self.data_info, self.clf)
            # make boundary
            self.boundary, self.class_nums = self._make_boundary(
                self.delta1, self.delta2)
            self.is_fit = True
            self.data_info_made = True
            if _print == True:
                print(
                    f"Found downsampled solution by removing {best_num_points_removed} number of points")

    
    def _check_and_optimise_data(self, data_info, best_loss, best_num_points_removed, _plot=False, _print=False):
        if self.check_if_solvable(data_info) == True:
            res = self._fit(data_info, _plot=_plot, _print=_print)
            # add penalty to the loss
            if data_info['num_reduced'] != 0:
                res['loss'] += data_info['alpha']*data_info['num_reduced']
            if best_loss == None:  # first result found so far
                best_loss = res['loss']
                best_num_points_removed = data_info['num_reduced']
                self._save_as_best(res, data_info)
            else:
                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    best_num_points_removed = data_info['num_reduced']
                    self._save_as_best(res, data_info)
            found = True
        else:
            found = False
        return best_loss, best_num_points_removed, found


    
    def _save_as_best(self, results, data_info):
        self.data_info = data_info
        self.delta1 = results['delta1']
        self.delta2 = results['delta2']
        self.solution_possible = results['solution_possible']
        self.solution_found = results['solution_found']
        print(self.delta1, self.delta2)

    def try_fit_downsample(self, X, y, costs):    
        # Make data_info - R_ests, D, etc.
        data_info = self.get_data_info(
            X, y, self.clf, costs, _print=False)
        
        return self.check_if_solvable(self, data_info), data_info

