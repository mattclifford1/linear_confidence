'''
scikit-learn style class to fit deltas
'''
import random
import multiprocessing

import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

import deltas.plotting.plots as plots
from deltas.model import base
 

class downsample_deltas(base.base_deltas):
    '''
    Downsample the dataset (randomly) until we find a good solution
    '''

    def __init__(self, clf, *args, **kwargs):
        super().__init__(clf, *args, **kwargs)

    def check_if_solvable(self, data_info):
        return downsample_deltas.check_if_solvable_static(data_info, self.contraint_func)

    @staticmethod
    def check_if_solvable_static(data_info, contraint_func):
        '''check the constraint to see if we have a viable solution'''
        if contraint_func(1, 1, data_info) <= 0:
            return True
        else:
            return False

    @staticmethod
    def random_downsample_data(X, y):
        '''randomly downsample the dataset'''
        # split into each class
        x1 = X[y == 0, :]
        x2 = X[y == 1, :]
        y1 = y[y == 0]
        y2 = y[y == 1]
        # downsample each class
        _x1, _y1, num_reduced1 = downsample_deltas.random_downsample_class(
            x1, y1)
        _x2, _y2, num_reduced2 = downsample_deltas.random_downsample_class(
            x2, y2)
        # put data back together - don't need to shuffle as deltas algorithm isn't affected by this
        return np.concatenate([_x1, _x2], axis=0), np.concatenate([_y1, _y2], axis=0), num_reduced1, num_reduced2

    @staticmethod
    def random_downsample_class(X, y):
        ' given only one class'
        num_samples = len(y)
        num_down_to = random.randint(1, num_samples)
        num_reduced = num_samples - num_down_to
        # now downsample
        _X, _y = resample(X, y, n_samples=num_down_to)
        return _X, _y, num_reduced

    @staticmethod
    def _test_single(args):
        '''wrapper for trialing downsample for use with multiprocessing'''
        X, y, costs, alpha, num_runs, contraint_func, loss_func, delta2_from_delta1 = args
        losses = []
        data_infos = []
        all_results = []
        for _ in range(num_runs):
            # downsample
            _X, _y, num_reduced1, num_reduced2 = downsample_deltas.random_downsample_data(
                X, y)
            # see if we can fit deltas
            data_info = downsample_deltas.get_data_info(
                _X, _y, costs=costs, _print=False)
            data_info['num_reduced'] = num_reduced1 + num_reduced2
            data_info['alpha'] = alpha
            results = downsample_deltas.static_check_and_optimise(
                data_info, contraint_func, loss_func, delta2_from_delta1)
            if results != None:
                losses.append(results['loss'])
                data_infos.append(data_info)
                all_results.append(results)
        return losses, data_infos, all_results

    def fit(self, X, y, costs=(1, 1), alpha=1, max_trials=10000, force_downsample=False, parallel=True, _plot=False, _print=False):
        '''
        fit to downsampled datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            max_trials:       the number of downsampled datasets to try 
            force_downsample: try downsampling even if the original projection is solvable
        '''
        # check we don't already have solvable without downsampling
        data_info = self.get_data_info(X, y, self.clf, costs, _print=False)
        data_info['num_reduced'] = 0
        results = self._check_and_optimise_data(data_info)
        if _plot == True:
            print('Original Data')
            self._plot_data(data_info, self.clf)

        # now try as many random downsamples of the dataset as the budget allows
        losses = []
        data_infos = []
        all_results = []
        if results != None:
            losses.append(results['loss'])
            data_infos.append(data_info)
            all_results.append(results)

        if results == None or force_downsample == True:
            # pre project data
            if hasattr(self.clf, 'get_projection'):
                X_projected = self.clf.get_projection(X)
            else:
                raise AttributeError(
                    f"Classifier {self.clf} needs 'get_projection' method")

            if parallel == True:
                n_cpus = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=n_cpus) as pool:
                    # add an argument of how many trials for each worker to do
                    # runs each worker does (counters the overhead of spawning a process)
                    num_runs = 100
                    scaled_trials = max_trials//num_runs
                    args = [[X_projected, y, costs, alpha, num_runs, self.contraint_func, self.loss_func, self.delta2_from_delta1]] * \
                        scaled_trials
                    trials = list(tqdm(pool.imap_unordered(self._test_single, args),
                                  total=scaled_trials, desc='Trying random downsampling deltas'))
                # now merge all the results together
                for result in trials:
                    for i in range(len(result[0])):
                        # results returned in format: losses, data_infos, all_results
                        losses.append(result[0][i])
                        data_infos.append(result[1][i])
                        all_results.append(result[2][i])

            else:
                for _ in tqdm(range(max_trials), desc='Trying random downsampling deltas', leave=True):
                    # downsample
                    _X, _y, num_reduced1, num_reduced2 = self.random_downsample_data(
                        X_projected, y)
                    # see if we can fit deltas
                    data_info = self.get_data_info(
                        _X, _y, costs=costs, _print=False)
                    data_info['num_reduced'] = num_reduced1 + num_reduced2
                    data_info['alpha'] = alpha
                    results = self._check_and_optimise_data(data_info)
                    if results != None:
                        losses.append(results['loss'])
                        data_infos.append(data_info)
                        all_results.append(results)

        else:
            if _print == True:
                print(
                    "Original dataset is solvable so not downsampling, set 'force_downsample' to 'True' to try and find a lower loss via downsampling anyway")

        # finished search, now make new boundary if we found a solution
        if len(losses) == 0:
            if _print == True:
                print('Unable to find result with downsample, increase the budget')
            self.is_fit = False
        else:
            best_ind = np.argmin(losses)
            self._save_as_best(all_results[best_ind], data_infos[best_ind])

            if _print == True:
                print(
                    f'With budget {max_trials} have found {len(losses)} viable downsampled solutions')
            # make boundary
            self.boundary, self.class_nums = self._make_boundary(
                self.delta1, self.delta2)
            self.is_fit = True
            if _print == True:
                print(
                    f"Best solution found by removing {self.data_info['num_reduced']} data points")
            if _plot == True:
                print('Downsampled data:')
                plots.deltas_projected_boundary(
                    self.delta1, self.delta2, self.data_info)
        return self

    @staticmethod
    def static_check_and_optimise(data_info, contraint_func, loss_func, delta2_from_delta1):
        '''return optim results, will be None if not solvable'''
        if downsample_deltas.check_if_solvable_static(data_info, contraint_func) == True:
            res = downsample_deltas._optimise(data_info,
                                              loss_func,
                                              contraint_func,
                                              delta2_from_delta1,
                                              _plot=False,
                                              _print=False)
            # add penalty to the loss
            if res != None and data_info['num_reduced'] != 0:
                res['loss'] += data_info['alpha']*data_info['num_reduced']
        else:
            res = None
        return res

    def _check_and_optimise_data(self, data_info):
        return self.static_check_and_optimise(data_info,
                                              self.contraint_func,
                                              self.loss_func,
                                              self.delta2_from_delta1)

    def _save_as_best(self, results, data_info):
        self.data_info = data_info
        self.data_info_made = True
        self.delta1 = results['delta1']
        self.delta2 = results['delta2']
        self.solution_possible = results['solution_possible']
        self.solution_found = results['solution_found']

    def try_fit_downsample(self, X, y, costs):
        # Make data_info - R_ests, D, etc.
        data_info = self.get_data_info(
            X, y, self.clf, costs, _print=False)

        return self.check_if_solvable(self, data_info), data_info
