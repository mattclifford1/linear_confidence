'''
scikit-learn style class to fit deltas with super set learning (SSL) dataset when not separable
'''
import random
import multiprocessing

import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

import deltas.plotting.plots as plots
from deltas.model import base, downsample

import time


class SSL_deltas(downsample.downsample_deltas):
    '''
    Super Set Learning (SSL) on deltas data
    '''

    def __init__(self, clf, *args, superset_classes=(0, 1, 2), **kwargs):
        self.superset_classes = superset_classes
        super().__init__(clf, *args, **kwargs)

    @staticmethod
    def random_swap_classes(X, y, superset_classes):
        '''randomly select classes from superset the dataset'''
        # split into each class
        x1 = X[y == 0, :]
        x2 = X[y == 1, :]
        y1 = y[y == 0]
        y2 = y[y == 1]
        # SSL each class
        superset = set(superset_classes)
        _x1, _y1 = SSL_deltas.random_swap_class(
            # x1, y1, {2})
            x1, y1, superset-{0})
        _x2, _y2 = SSL_deltas.random_swap_class(
            # x2, y2, {2})
            x2, y2, superset-{1})

        return np.concatenate([_x1, _x2], axis=0), np.concatenate([_y1, _y2], axis=0)

    @staticmethod
    def random_swap_class(X, y, superset_classes):
        ' given only one class'
        _y = np.copy(y)
        superlist = list(superset_classes)
        num_samples = len(y)
        num_change = random.randint(1, num_samples-1)
        single_change = num_change//len(superlist)
        for clas in superlist:
            mask = np.random.permutation(np.array([True]*single_change + [False]*(y.size-single_change))).reshape(y.shape)
            _y[mask] = clas # update random elements
        return X, _y
    
    @staticmethod
    def _test_single(args, disable_tqdm=True):
        # n = 1000000
        # t0 = time.time()
        # for i in tqdm(range(n)):
        #     # downsample
        #     _y = SSL_deltas.random_swap_classes(
        #         args['y'], args['superset_classes'])
        #     # see if we can fit deltas
        #     data_info = base.base_deltas.get_data_info(
        #         args['X'], _y, costs=args['costs'], _print=False)

        #     # results = downsample.downsample_deltas.static_check_and_optimise(
        #     #     data_info, args['contraint_func'], args['loss_func'], args['delta2_from_delta1'], args['grid_search'])

        # t1 = time.time()
        # print(t1-t0)

        '''wrapper for trialing downsample for use with multiprocessing'''
        losses = []
        data_infos = []
        all_results = []
        for _ in tqdm(range(args['num_runs']), desc='Trying random SSL deltas', leave=False, disable=disable_tqdm):
            # SSL
            _X, _y = SSL_deltas.random_swap_classes(args['X'], args['y'], args['superset_classes'])
            # see if we can fit deltas
            data_info = base.base_deltas.get_data_info(_X, _y, costs=args['costs'], 
                                                       _print=False, supports=False)
            data_info['alpha'] = args['alpha']
            data_info['prop_penalty'] = args['prop_penalty']

            # print(f"{data_info['N1']+data_info['N2']}, {data_info['N1']}, {data_info['N2']}")

            results = downsample.downsample_deltas.static_check_and_optimise(
                data_info, args['contraint_func'], args['loss_func'], args['delta2_from_delta1'], args['grid_search'])
            if results != None:
                # print('found')
                losses.append(results['loss'])
                data_infos.append(data_info)
                all_results.append(results)
        return losses, data_infos, all_results

    def fit(self, X, y, costs=(1, 1), alpha=1, prop_penalty=True, max_trials=10000, force_SSL=False, parallel=True, grid_search=True, _plot=False, _print=False):
        '''
        fit to SSL datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            prop_penalty:     scale penality per class based on proportion of samples removed
            max_trials:       the number of downsampled datasets to try 
            force_SSL:        try SSL even if the original projection is solvable
        '''
        # check we don't already have solvable without downsampling
        data_info = self.get_data_info(X, y, self.clf, costs, _print=False)
        results = self._check_and_optimise_data(data_info)
        if _plot == True:
            print('Original Data')
            self._plot_data(data_info, self.clf)

        # now try as many random SSL of the dataset as the budget allows
        losses = []
        data_infos = []
        all_results = []
        if results != None:
            losses.append(results['loss'])
            data_infos.append(data_info)
            all_results.append(results)

        if results == None or force_SSL == True:
            SSLed = True
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
                                'grid_search': grid_search,
                                'superset_classes': self.superset_classes}
                    args = [arg_dict] * scaled_trials
                    trials = list(tqdm(pool.imap_unordered(self._test_single, args),
                                       total=scaled_trials, desc=f'Trying random SSL deltas (multiprocessing batches of {num_runs})', leave=False))
            else:
                arg_dict = {'X': X_projected,
                            'y': y,
                            'costs': costs,
                            'alpha': alpha,
                            'prop_penalty': prop_penalty,
                            'num_runs': max_trials,
                            'contraint_func': self.contraint_func,
                            'loss_func': self.loss_func,
                            'delta2_from_delta1': self.delta2_from_delta1,
                            'grid_search': grid_search,
                            'superset_classes': self.superset_classes}
                trials = [self._test_single(arg_dict, disable_tqdm=False)]

            # now merge all the results together
            for result in trials:
                for i in range(len(result[0])):
                    # results returned in format: (losses, data_infos, all_results)
                    losses.append(result[0][i])
                    data_infos.append(result[1][i])
                    all_results.append(result[2][i])
        else:
            SSLed = False
            if _print == True:
                print(
                    "Original dataset is solvable so no SSL, set 'force_SSL' to 'True' to try and find a lower loss via SSL anyway")

        # finished search, now make new boundary if we found a solution
        if len(losses) == 0:
            if _print == True:
                print('Unable to find result with SSL, increase the max_trials')
            self.is_fit = False
        else:
            best_ind = np.argmin(losses)
            self._save_as_best(all_results[best_ind], data_infos[best_ind])

            if _print == True and SSLed == True:
                print(
                    f'With budget {max_trials} have found {len(losses)} viable SSL solutions')
            # make boundary
            self.boundary, self.class_nums = self._make_boundary(
                self.delta1, self.delta2)
            self.is_fit = True
            if _plot == True:
                if SSLed == True:
                    print('SSL Data:')
                else:
                    print('Original Data:')
                plots.deltas_projected_boundary(
                    self.delta1, self.delta2, self.data_info)
        return self
