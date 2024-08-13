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
from deltas.misc.use_two import USE_GLOBAL_R


class downsample_deltas(base.base_deltas):
    '''
    Downsample the dataset (randomly) until we find a good solution
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y,
            costs=(1, 1),
            alpha=1,
            prop_penalty=True,
            continuous_slacks=False,
            method='supports-prop-update_mean',
            max_trials=1000,
            force_downsample=False,
            parallel=True,
            clf=None,
            grid_search=True,
            _plot=False,
            _print=False,
            save_file=None,
            diagram=False):
        '''
        fit to downsampled datasets, then pick the lowest loss
            alpha:            the penalty value on the loss for removing points
            prop_penalty:     scale penality per class based on proportion of samples removed
            continuous_slacks: use continuous slacks in the optimisation (False = Binary)
            max_trials:       the number of downsampled datasets to try 
            force_downsample: try downsampling even if the original projection is solvable
            method:           which method of downsampling to use from: ['supports', 'supports-update_mean', 'supports-prop', 'supports-prop-update_mean', 'random']
        '''
        if not isinstance(clf, type(None)):
            if not hasattr(clf, 'get_projection'):
                raise AttributeError(
                    f"Classifier {clf} needs 'get_projection' method")
            self.clf = clf
        # check method is supported
        methods_supported = ['supports',
                             'supports-update_mean',
                             'supports-prop',
                             'supports-prop-update_mean',
                             'supports-prop-update_mean-margin_only',
                             'random']
        # if method not in methods_supported:
        #     raise ValueError(f'method must be one of {methods_supported} not {method}')

        # check we don't already have solvable without downsampling
        data_info = self.get_data_info(X, y, self.clf, costs, _print=False)
        data_info['num_reduced'] = 0
        results = self._check_and_optimise_data(data_info)
        if _plot == True:
            print('Original Data')
            self._plot_data(data_info, self.clf, save_file=save_file, diagram=diagram)

        # now try as many random downsamples of the dataset as the budget allows
        if 'supports' in method:
            max_trials = min(len(y)//2, max_trials)

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
                num_runs_per_batch = max(min(100, max_trials//n_cpus), 1)
                num_runs = num_runs_per_batch
            else:
                num_runs = max_trials
            arg_dict = {'X': X_projected,
                        'y': y,
                        'costs': costs,
                        'alpha': alpha,
                        'prop_penalty': prop_penalty,
                        'continuous_slacks': continuous_slacks,
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
                    print(
                        'Unable to find result with downsample, increase the max_trials')
                else:
                    print(
                        'Dataset projection incompatible with deltas downsample supports method')
            self.is_fit = False
            self.loss = np.inf
        else:
            best_ind = np.argmin(losses)
            self.loss = np.min(losses)
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
                    self.delta1, self.delta2, self.data_info, save_file=save_file)
        return self
    
    def score(self, *args, **kwargs):
        return self.loss

    def check_if_solvable(self, data_info):
        return downsample_deltas.check_if_solvable_static(data_info, self.contraint_func, self.delta2_from_delta1)

    @staticmethod
    def check_if_solvable_static(data_info, contraint_func, delta2_from_delta1):
        '''check the constraint to see if we have a viable solution'''
        # quick check if Rs are over lapping - can return quicker if they are
        if data_info['empirical R1'] + data_info['empirical R2'] > data_info['empirical D']:
            return False
        
        # make sure no diving by zero
        if USE_GLOBAL_R == True:
            if data_info['empirical R1'] == 0 or data_info['empirical R2'] == 0:
                return False
        
        # make sure instances of both classes
        if data_info['N1'] == 0 or data_info['N2'] == 0:
            return False
        
        # now check constraint/ if minimum error terms overlap too
        highest_delta = 0.99999999999999999999999999999
        
        try: # wrap to find out if solvable sometimes divides by zero
            # actual check
            contraint_val = contraint_func(highest_delta, delta2_from_delta1(
                highest_delta, data_info), data_info)
            
            # # loose check
            # contraint_val = contraint_func(
            #     highest_delta, highest_delta, data_info)
            
        except ZeroDivisionError:
            return False
        
        # see if solvable
        if contraint_val <= 0:
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
                                 update_means=False,
                                 remove_margin_only=False):
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

        # use for continuous slacks calc
        original_means = (np.mean(x1, axis=0), np.mean(x2, axis=0)) 
        removed_points_from_mean = [[], []]

        # use original means in supports calc or not
        if update_means == True:
            ms = [np.mean(x1, axis=0), np.mean(x2, axis=0)]
        else:
            ms = [None, None] # this will tell it to calculate the mean based on downsampled data

        num_reduced = [0, 0]

        while sum(num_reduced) < num_to_reduce:
            # TODO: make more effecient by using slicing views instead of delete which creates a new arrays

            # loop over each class
            for order in [[0, 1], [1, 0]]:  # need both current and other ind for proportion calc
                # make sure we have some data points left
                if ys[order[0]].shape[0] > 1:
                    # remove to keep the correct class ratios 
                    while ys[order[0]].shape[0]/ys[order[1]].shape[0] >= num_originals[order[0]]/num_originals[order[1]]:
                        # find the curent support
                        if remove_margin_only == False:
                            ind = data_utils.get_support_of_class_ind(Xs[order[0]], ms[order[0]])
                        else:
                            ind = data_utils.get_support_of_class_ind_margin_only(Xs[order[0]], 
                                                                                  Xs[order[1]],
                                                                                  ms[order[1]])

                        # add support to be removed info to slack continuous vars
                        removed_points_from_mean[order[0]].append(
                            np.abs(Xs[order[0]][ind] - original_means[order[0]])[0])
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
                        # break if we've reached the reduction limit
                        if sum(num_reduced) == num_to_reduce:
                            break


                    # break if we've reached the reduction limit
                    if sum(num_reduced) == num_to_reduce:
                        break

        # slack var continuous values to numpy array
        for i, _ in enumerate(removed_points_from_mean):
            removed_points_from_mean[i] = np.array(removed_points_from_mean[i])
        
        # print for dev to make sure the proportions looks good for prop method
        # total_orig = num_originals[0] + num_originals[1]
        # print([num_originals[0]/total_orig, num_originals[1]/total_orig])
        # total_new = ys[0].shape[0] + ys[1].shape[0]
        # print([ys[0].shape[0]/total_new, ys[1].shape[0]/total_new])
        # print('')
        new_X = np.concatenate([Xs[0], Xs[1]], axis=0)
        new_y = np.concatenate([ys[0], ys[1]], axis=0)
        return new_X, new_y, num_reduced[0], num_reduced[1], removed_points_from_mean


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
                if 'margin_only' in args['downsample_method']:
                    remove_margin_only = True
                else:
                    remove_margin_only = False
                # see where we are at in terms of workers
                num_to_reduce = i + args['order_in_queue']*args['num_runs']
                # remove correct amount of supports
                _X, _y, num_reduced1, num_reduced2, removed_points_from_mean = downsample_deltas.supports_downsample_data(
                    args['X'], args['y'], 
                    num_to_reduce, 
                    remove_method=remove_method, 
                    update_means=update_means,
                    remove_margin_only=remove_margin_only)
            

            # see if we can fit deltas
            data_info = base.base_deltas.get_data_info(
                _X, _y, costs=args['costs'], _print=False, supports=False)
            data_info['num_reduced'] = num_reduced1 + num_reduced2
            data_info['num_reduced_1'] = num_reduced1
            data_info['num_reduced_2'] = num_reduced2
            data_info['alpha'] = args['alpha']
            data_info['prop_penalty'] = args['prop_penalty']
            data_info['continuous_slacks'] = args['continuous_slacks']
            data_info['removed_points_from_mean'] = removed_points_from_mean
            data_info['removed_points_from_mean_1'] = removed_points_from_mean[0]
            data_info['removed_points_from_mean_2'] = removed_points_from_mean[1]

            results = downsample_deltas.static_check_and_optimise(
                data_info, args['contraint_func'], args['loss_func'], args['delta2_from_delta1'], args['grid_search'])
            if results != None:
                # print('solution found')
                losses.append(results['loss'])
                data_infos.append(data_info)
                all_results.append(results)
        return losses, data_infos, all_results
    

    @staticmethod
    def static_check_and_optimise(data_info, contraint_func, loss_func, delta2_from_delta1, grid_search=True):
        '''return optim results, will be None if not solvable'''
        # solve if solvable
        solvable = downsample_deltas.check_if_solvable_static(data_info, contraint_func, delta2_from_delta1)
        if solvable == True:
            res = downsample_deltas._optimise(data_info,
                                              loss_func,
                                              contraint_func,
                                              delta2_from_delta1,
                                              grid_search=grid_search,
                                              _plot=False,
                                              _print=False)
            # add penalty to the loss
            # first check what sort of penalty we are adding
            if 'num_reduced' in data_info.keys():
                num_reduced = data_info['num_reduced']
            else:
                num_reduced = 0
            if 'continuous_slacks' in data_info.keys():
                continuous_slacks = data_info['continuous_slacks']
            else:
                continuous_slacks = False

            # add the penalty
            if res != None and num_reduced != 0:
                if data_info['prop_penalty'] == True:
                    for i, c in enumerate([1, 2]):
                        # see if we have single or alpha per class
                        alpha = data_info['alpha']
                        if hasattr(data_info['alpha'], '__len__'):
                            if len(data_info['alpha']) == 2:
                                alpha = data_info['alpha'][i]
                        
                        # add proportional loss
                        if continuous_slacks == True:
                            slacks = np.sum(data_info[f'removed_points_from_mean_{c}'])
                        else: 
                            slacks = data_info[f'num_reduced_{c}']
                        res['loss'] += alpha * \
                            ( slacks / (data_info[f'N{c}']+data_info[f'num_reduced_{c}']))
                else:  # add regular loss
                    if continuous_slacks == True:
                        slacks = np.sum(
                            data_info['removed_points_from_mean_1']) + np.sum(data_info['removed_points_from_mean_2'])
                    else:
                        slacks = data_info['num_reduced']
                    res['loss'] += data_info['alpha']*slacks
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
