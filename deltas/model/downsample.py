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
        _X, _y = resample(X, y, n_samples=num_down_to, replace=False)
        return _X, _y, num_reduced

    @staticmethod
    def _test_single(args, disable_tqdm=True):
        '''wrapper for trialing downsample for use with multiprocessing'''
        losses = []
        data_infos = []
        all_results = []
        for _ in tqdm(range(args['num_runs']), desc='Trying random downsampling deltas', leave=False, disable=disable_tqdm):
            # downsample
            _X, _y, num_reduced1, num_reduced2 = downsample_deltas.random_downsample_data(
                args['X'], args['y'])
            # see if we can fit deltas
            data_info = downsample_deltas.get_data_info(
                _X, _y, costs=args['costs'], _print=False)
            data_info['num_reduced'] = num_reduced1 + num_reduced2
            data_info['num_reduced_1'] = num_reduced1
            data_info['num_reduced_2'] = num_reduced2
            data_info['alpha'] = args['alpha']
            data_info['prop_penalty'] = args['prop_penalty']

            results = downsample_deltas.static_check_and_optimise(
                data_info, args['contraint_func'], args['loss_func'], args['delta2_from_delta1'], args['grid_search'])
            if results != None:
                losses.append(results['loss'])
                data_infos.append(data_info)
                all_results.append(results)
        return losses, data_infos, all_results

    def fit(self, X, y, costs=(1, 1), alpha=1, prop_penalty=True, max_trials=10000, force_downsample=False, parallel=True, grid_search=True, _plot=False, _print=False):
        '''
        fit to downsampled datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            prop_penalty:     scale penality per class based on proportion of samples removed
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
            downsampled = True
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
                    arg_dict = {'X': X_projected,
                                'y': y,
                                'costs': costs,
                                'alpha': alpha,
                                'prop_penalty': prop_penalty,
                                'num_runs': num_runs,
                                'contraint_func': self.contraint_func,
                                'loss_func': self.loss_func,
                                'delta2_from_delta1': self.delta2_from_delta1,
                                'grid_search': grid_search}
                    args = [arg_dict] * scaled_trials
                    trials = list(tqdm(pool.imap_unordered(self._test_single, args),
                                       total=scaled_trials, desc=f'Trying random downsampling deltas (multiprocessing batches of {num_runs})', leave=False))
            else:
                arg_dict = {'X': X_projected,
                            'y': y,
                            'costs': costs,
                            'alpha': alpha,
                            'prop_penalty': prop_penalty,
                            'num_runs': max_trials,
                            'contraint_func': self.contraint_func,
                            'loss_func': self.loss_func,
                            'delta2_from_delta1': self.delta2_from_delta1}
                trials = [self._test_single(arg_dict, disable_tqdm=False)]
                                    # total=max_trials, desc='Trying random downsampling deltas'), leave=False)
            # now merge all the results together
            for result in trials:
                for i in range(len(result[0])):
                    # results returned in format: (losses, data_infos, all_results)
                    losses.append(result[0][i])
                    data_infos.append(result[1][i])
                    all_results.append(result[2][i])
        else:
            downsampled = False
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

            if _print == True and downsampled == True:
                print(
                    f'With budget {max_trials} have found {len(losses)} viable downsampled solutions')
            # make boundary
            self.boundary, self.class_nums = self._make_boundary(
                self.delta1, self.delta2)
            self.is_fit = True
            if _print == True and downsampled == True:
                print(
                    f"Best solution found by removing {self.data_info['num_reduced']} data points")
            if _plot == True:
                if downsampled == True:
                    print('Downsampled Data:')
                else:
                    print('Original Data:')
                plots.deltas_projected_boundary(
                    self.delta1, self.delta2, self.data_info)
        return self

    @staticmethod
    def static_check_and_optimise(data_info, contraint_func, loss_func, delta2_from_delta1, grid_search=True):
        '''return optim results, will be None if not solvable'''
        if downsample_deltas.check_if_solvable_static(data_info, contraint_func) == True:
            res = downsample_deltas._optimise(data_info,
                                              loss_func,
                                              contraint_func,
                                              delta2_from_delta1,
                                              grid_search=grid_search,
                                              _plot=False,
                                              _print=False)
            # add penalty to the loss
            if res != None and data_info['num_reduced'] != 0:
                if data_info['prop_penalty'] == True:
                    for i, c in enumerate([1, 2]):
                        # see if we have single or alpha per class
                        alpha = data_info['alpha']
                        if hasattr(data_info['alpha'], '__len__'):
                            if len(data_info['alpha']) == 2:
                                alpha = data_info['alpha'][i]
                        # add proportional loss
                        res['loss'] += alpha * \
                            (data_info[f'num_reduced_{c}'] /
                            (data_info[f'N{c}']+data_info[f'num_reduced_{c}']))
                else:
                    # add regular loss
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
