'''
scikit-learn style class to fit deltas with downsampling dataset when not separable
'''
import random
import multiprocessing

import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

import deltas.plotting.plots as plots
from deltas.model import base
from deltas.utils import data as data_utils


class downsample_deltas(base.base_deltas):
    '''
    Downsample the dataset (randomly) until we find a good solution
    '''
    def __init__(self, clf, *args, **kwargs):
        super().__init__(clf, *args, **kwargs)

    def check_if_solvable(self, data_info):
        return downsample_deltas.check_if_solvable_static(data_info, self.contraint_func, self.delta2_from_delta1)

    @staticmethod
    def check_if_solvable_static(data_info, contraint_func, delta2_from_delta1):
        '''check the constraint to see if we have a viable solution'''
        # quick check if Rs are over lapping - can return quicker if they are
        if data_info['empirical R1'] + data_info['empirical R2'] > data_info['empirical D']:
            return False
        # now check constraint/ if minimum error terms overlap too
        if contraint_func(1, delta2_from_delta1(1, data_info), data_info) <= 0:
            return True
        else:
            return False

    @staticmethod
    def random_downsample_data(X, y):
        '''randomly downsample the dataset'''
        # split into each class
        x1, x2, y1, y2 = data_utils.split_classes(X, y)
        
        # downsample each class
        def random_downsample_class(X, y):
            ' given only one class'
            num_samples = len(y)
            num_down_to = random.randint(1, num_samples)
            num_reduced = num_samples - num_down_to
            # now downsample
            _X, _y = resample(X, y, n_samples=num_down_to, replace=False)
            return _X, _y, num_reduced
        _x1, _y1, num_reduced1 = random_downsample_class(x1, y1)
        _x2, _y2, num_reduced2 = random_downsample_class(x2, y2)

        # put data back together - don't need to shuffle as deltas algorithm isn't affected by this
        return np.concatenate([_x1, _x2], axis=0), np.concatenate([_y1, _y2], axis=0), num_reduced1, num_reduced2

    @staticmethod
    def supports_downsample_data(X, y, 
                                 num_to_reduce, 
                                 remove_method='equal', 
                                 update_means=False):
        '''downsample the dataset one support at a time'''
        # check support method selected
        supported_methods = ['equal', 'proportional']
        if remove_method not in supported_methods:
            raise ValueError(f'Supports remove method needs to be one of {supported_methods} not {remove_method}')

        # split into each class
        x1, x2, y1, y2 = data_utils.split_classes(X, y)
        Xs = [x1, x2]
        ys = [y1, y2]
        num_originals = (y1.shape[0], y2.shape[0])

        # use original means in supports calc
        if update_means == True:
            ms = [np.mean(x1, axis=0), np.mean(x2, axis=0)]
        else:
            ms = [None, None]

        num_reduced = [0, 0]

        while sum(num_reduced) < num_to_reduce:
            # TODO: make more effecient by using slicing views instead of delete which creates a new arrays

            # loop over each class
            for order in [[0, 1], [1, 0]]:
                # make sure we have some data points left
                if ys[order[0]].shape[0] > 1:
                    # remove to keep the correct class ratios 
                    while ys[order[0]].shape[0]/ys[order[1]].shape[0] >= num_originals[order[0]]/num_originals[order[1]]:
                        # find the curent support
                        ind = data_utils.get_support_of_class_ind(Xs[order[0]], ms[order[0]])
                        # delete the support found
                        Xs[order[0]] = np.delete(Xs[order[0]], [ind])
                        ys[order[0]] = np.delete(ys[order[0]], [ind])
                        # correct the numpy array dims
                        if len(Xs[order[0]].shape) == 1:
                            Xs[order[0]] = np.expand_dims(Xs[order[0]], axis=1)

                        # keep count of the number reduced for this class
                        num_reduced[order[0]] += 1

                        # break if we dont care about class ratios
                        if remove_method == 'equal':
                            break
                        # break if weve reached the reduction limit
                        if sum(num_reduced) == num_to_reduce:
                            break
                    # break if weve reached the reduction limit
                    if sum(num_reduced) == num_to_reduce:
                        break
        
        # print for dev to make sure the proportions looks good for prop method
        # total_orig = num_originals[0] + num_originals[1]
        # print([num_originals[0]/total_orig, num_originals[1]/total_orig])
        # total_new = ys[0].shape[0] + ys[1].shape[0]
        # print([ys[0].shape[0]/total_new, ys[1].shape[0]/total_new])
        # print('')

        return np.concatenate([Xs[0], Xs[1]], axis=0), np.concatenate([ys[0], ys[1]], axis=0), num_reduced[0], num_reduced[1]


    @staticmethod
    def _test_single(args, disable_tqdm=True):
        '''wrapper for trialing downsample for use with multiprocessing'''
        losses = []
        data_infos = []
        all_results = []
        for i in tqdm(range(args['num_runs']), desc=f"Trying {args['downsample_method']} downsampling deltas", leave=False, disable=disable_tqdm):
            # downsample
            if args['downsample_method'] == 'random':
                _X, _y, num_reduced1, num_reduced2 = downsample_deltas.random_downsample_data(args['X'], args['y'])
            elif 'supports' in args['downsample_method']:
                # get type of supports removeal
                if 'prop' in args['downsample_method']:
                    remove_method = 'proportional'
                else:
                    remove_method = 'equal'
                if 'update_mean' in args['downsample_method']:
                    update_means = True
                else:
                    update_means = False
                # see where we are at in terms of workers
                num_to_reduce = i + args['order_in_queue']*args['num_runs']
                # remove correct amount of supports
                _X, _y, num_reduced1, num_reduced2 = downsample_deltas.supports_downsample_data(
                    args['X'], args['y'], num_to_reduce, remove_method=remove_method, update_means=update_means)
            

            # see if we can fit deltas
            data_info = base.base_deltas.get_data_info(
                _X, _y, costs=args['costs'], _print=False, supports=False)
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
    

    def fit(self, X, y, 
            costs=(1, 1), 
            alpha=1, 
            prop_penalty=True, 
            method='random', 
            max_trials=10000, 
            force_downsample=False, 
            parallel=True, 
            grid_search=True, 
            _plot=False, 
            _print=False):
        '''
        fit to downsampled datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            prop_penalty:     scale penality per class based on proportion of samples removed
            max_trials:       the number of downsampled datasets to try 
            force_downsample: try downsampling even if the original projection is solvable
            method:           which method of downsampling to use from: ['supports', 'supports-update_mean', 'supports-prop', 'supports-prop-update_mean', 'random']
        '''
        # check method is supported
        methods_supported = ['supports',
                             'supports-update_mean',
                             'supports-prop',
                             'supports-prop-update_mean',
                             'random']
        if method not in methods_supported:
            raise ValueError(f'method must be one of {methods_supported} not {method}')

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
            
            # only do as many trials as data points for non random methods
            methods_supported.remove('random')
            if method in methods_supported:
            # if method != 'random':
                max_able = X.shape[0] - 2
                if max_trials >= max_able:
                    max_trials = max_able
                    support_max_hit = True
                else:
                    support_max_hit = False

            # pre project data for efficiency
            if hasattr(self.clf, 'get_projection'):
                X_projected = self.clf.get_projection(X)
            else:
                raise AttributeError(
                    f"Classifier {self.clf} needs 'get_projection' method")
            
            # set up args
            if parallel == True:
                n_cpus = multiprocessing.cpu_count()
                # see how many to run per batch (optimise for n_cpus)
                num_runs_per_batch = min(100, max_trials//n_cpus)
                num_runs = num_runs_per_batch
            else:
                num_runs = max_trials
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
                        'downsample_method': method,
                        'order_in_queue': 0}

            if parallel == True:
                with multiprocessing.Pool(processes=n_cpus) as pool:
                    # add an argument of how many trials for each worker to do
                    # runs each worker does (counters the overhead of spawning a process)
                    scaled_trials = max_trials//num_runs

                    # get all the args per worker
                    args = []
                    for i in range(scaled_trials):
                        args.append(arg_dict.copy())
                        # order the jobs to each work knows what to compute
                        args[i]['order_in_queue'] = i

                    # run all workers
                    trials = list(tqdm(pool.imap_unordered(self._test_single, args),
                                       total=scaled_trials, desc=f'Trying random downsampling deltas (multiprocessing batches of {num_runs})', leave=False))
            else:
                trials = [self._test_single(arg_dict, disable_tqdm=False)]

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
                if method not in methods_supported or support_max_hit == False:
                    print('Unable to find result with downsample, increase the max_trials')
                else:
                    print('Dataset projection incompatible with deltas downsample supports method')
            self.is_fit = False
        else:
            best_ind = np.argmin(losses)
            self._save_as_best(all_results[best_ind], data_infos[best_ind])

            if _print == True and downsampled == True:
                print(
                    f'Budget {max_trials} found {len(losses)} viable downsampled solutions')
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
        if downsample_deltas.check_if_solvable_static(data_info, contraint_func, delta2_from_delta1) == True:
            res = downsample_deltas._optimise(data_info,
                                              loss_func,
                                              contraint_func,
                                              delta2_from_delta1,
                                              grid_search=grid_search,
                                              _plot=False,
                                              _print=False)
            # add penalty to the loss
            if 'num_reduced' in data_info.keys():
                num_reduced = data_info['num_reduced']
            else:
                num_reduced = 0
            if res != None and num_reduced != 0:
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
