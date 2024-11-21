'''
scikit-learn style class to fit deltas in the overlapping case
'''
import numpy as np
from matplotlib import pyplot as plt

import deltas.utils.projection as projection
import deltas.utils.radius as radius
from deltas.misc.use_two import USE_TWO


class deltas:
    def __init__(self, clf=None, dim_reducer=None, dev=False):
        if not hasattr(clf, 'get_projection'):
            raise AttributeError(
                f"Classifier {clf} needs 'get_projection' method")
        self.clf = clf
        self.dim_reducer = dim_reducer
        self.dev = dev
        self._setup()
        

    def _setup(self):
        self.data_info_made = False
        self.data_info_made = False
        self.is_fit = False

    def fit(self, X, y, 
            costs=(1, 1), 
            _plot=False, 
            only_furtherest_k=False, 
            loss_type='min',
            **kwargs):
        self.data_info = data_info(X, y, self.clf)
        self.data_info_made = True
        self.check_solvable()
        self.costs = costs
        self.only_furtherest_k = only_furtherest_k
        self.loss_type = loss_type
        self.optimise(_plot=_plot, **kwargs)
        self.is_fit = True
        return self
    
    def check_solvable(self):
        ''' check if the problem is solvable with data'''
        if self.data_info.N1 == 0 or self.data_info.N2 == 0:
            raise ValueError("One class has no data points")
        # if the means are too close
        if self.data_info.D_emp < (self.data_info.min_conc_1 + self.data_info.min_conc_2):
            raise ValueError("Means are too close")
    
    def _check_fit(self):
        if not self.is_fit:
            raise AttributeError("Call .fit(X, y) first")
        
    def _check_data_info_made(self):
        if self.data_info_made == False:
            raise AttributeError("data_info not made")
        
    def check_bias_valid(self, bias):
        m1, m2 = self.data_info('emp_xp1'), self.data_info('emp_xp2')
        if bias < m1 and bias < m2:
            return False
        if bias > m1 and bias > m2:
            return False

        for cls in [1, 2]:
            mean = self.data_info(f'emp_xp{cls}')
            d_bias = np.abs(bias - mean)
            if d_bias < self.data_info(f'min_conc_{cls}'):
                return False
        return True
    
    def get_valid_optimisation_range(self):
        ''' get the valid range for the optimisation taking into account min conc '''
        m1, m2 = self.data_info('emp_xp1'), self.data_info('emp_xp2')
        if m1 < m2:
            r1 = m1 + self.data_info('min_conc_1')
            r2 = m2 - self.data_info('min_conc_2')
        else:
            r1 = m1 - self.data_info('min_conc_1')
            r2 = m2 + self.data_info('min_conc_2')
        return r1, r2
    
    def get_valid_linspace(self, r1, r2, res):
        ''' get a linspace with any invalid points removed'''
        line = np.linspace(r1, r2, res)
        # remove the invalid points
        rm_inds = []
        for i, bias in enumerate(line):
            if self.check_bias_valid(bias) == False:
                rm_inds.append(i)
        line = np.delete(line, rm_inds)
        return line
    
    def optimise(self, res=1000, _plot=False, **kwargs):
        ''' optimise the deltas and return the best bias using linspace min'''
        if self.data_info_made == False:
            raise AttributeError("data_info not made -- dev error")
        r1, r2 = self.get_valid_optimisation_range()
        line = self.get_valid_linspace(r1, r2, res)
        # remove the invalid points
        if self.check_bias_valid(line[0]) == False:
            np.delete(line, 0)
        if self.check_bias_valid(line[-1]) == False:
            np.delete(line, -1)
        
        # get all loss values
        losses = [self.get_loss(bias) for bias in line]
        
        self.best_bias = line[np.argmin(losses)]
        # plot
        if _plot == True:
            plt.plot(line, losses)
            plt.xlabel("Bias")
            plt.ylabel("Loss")
            plt.show()

    def _single_class_loss(self, error, delta):
        loss = error*(1-delta) + delta
        return loss
       

    def get_loss(self, bias):
        ''' get the loss as a specific bias point'''
        self._check_data_info_made()
        if self.check_bias_valid(bias) == False:
            return None
        
        # get loss for each class
        total_loss = 0
        for cls in [1, 2]:
            # get the error term
            errors, points, ks = self.get_generalisation_error(bias, cls=cls)
            losses = []
            # get the loss for each point
            for error, point in zip(errors, points):
                # get the delta for this class
                delta = self.deltas_from_bias(bias, point, cls)
                # get the loss for this class
                losses.append(self._single_class_loss(error, delta))
            
            if self.loss_type == 'mean':
                total_loss += np.mean(losses)
            elif self.loss_type == 'max':
                ind = np.argmax(losses)
                total_loss += losses[np.argmax(losses)]
            elif self.loss_type == 'min':
                ind = np.argmin(losses)
                total_loss += losses[np.argmin(losses)]
            else:
                raise AttributeError("loss_type must be 'mean', 'max' or 'min'")



            if self.dev == True:
                print(f'min k cls {cls}: {ks[ind]}/{len(ks)} bias: {bias}')

        return total_loss

    def deltas_from_bias(self, bias, point, cls=1):
        ''' get the deltas i for class i at a specific bias point
        point is the training point we are using to calculate from
        same as eq. 8 in the original paper but with B = distance_bias - distance_point'''
        self._check_data_info_made()
        # get vars we need
        mean = self.data_info(f'emp_xp{cls}')
        R = self.data_info(f'R{cls}_emp')
        N = self.data_info(f'N{cls}')

        # get the distance of bias point to mean
        d_bias = np.abs(bias - mean)
        # get the distance of training point to mean
        d_point = np.abs(point - mean)

        # check bias is in a valid place
        if d_bias < self.data_info(f'min_conc_{cls}'):
            raise ValueError(f"Bias point is too close to mean {cls}")

        # use two or not
        factor = self.get_use_two()
        # factor = 1

        # inside the exponent
        B = d_bias - d_point

        inside = np.square(( B / ((factor*R) / (np.sqrt(N))) ) - 2) / 2
        # get delta
        delta = np.exp(-inside)

        if self.dev == True:
            # print(f'{d_bias} = {d_point} + error = {d_bias - d_point}')
            # recover the bias term to check all is working
            err = radius.error_upper_bound(R, N, delta)
            # err = ((factor*R) / (np.sqrt(N))) * (2 + np.sqrt(2*np.log(1/delta)))
            # print(f'error = {err}')

        return delta

    def get_generalisation_error(self, X_t, cls=1):
        '''X_t is the test data points or current bias term
        and we want to give the error term from the empirical mean before
        taking into account the error around the empirical mean vs expectation'''
        self._check_data_info_made()
        # get vars we need
        mean = self.data_info(f'emp_xp{cls}')
        N = self.data_info(f'N{cls}')
        d_train_orig = self.data_info(f'd_{cls}')

        # add the min concentration inequality error onto the dists as thats the dist we care about
        d_train = d_train_orig + self.data_info(f'min_conc_{cls}')
        # get the distance of test point to mean
        d_test = np.abs(X_t - mean)
        # where the test point vs train points
        d_comp = d_test > d_train

        # if the test point is further away than the furthest train point
        if d_comp.all() == True:
            # old error term -- is the beyond all train points
            k_furthest = [1]
        else:
            # argmin will give the index of the first False value
            k_furthest = [N + 1 - np.argmin(d_comp)]

        # use all points closer as well to evaluate the error
        if self.only_furtherest_k == False:
            for i in range(k_furthest[0], N):
                k_furthest.append(i+1)
        # get the errors and points used
        errors = []
        points = []
        # equation 1 (but new version)
        for k in k_furthest:
            errors.append(k/(N+1))
            points.append(
                self.get_training_point_from_k_furthest(X_t, k, cls=cls))
            

        return errors, points, k_furthest
    
    def get_training_point_from_k_furthest(self, X_t, k_furthest, cls=1):
        ''' get the training point location that is k_furthest away from 
        the test point -- used when getting the point accociated with loss'''
        # get vars we need
        mean = self.data_info(f'emp_xp{cls}')
        N = self.data_info(f'N{cls}')
        d_train_orig = self.data_info(f'd_{cls}')

        ind = N - k_furthest
        if ind == -1:  # point closer than all training points
            dist_add = 0
        else:
            dist_add = d_train_orig[ind]
        # get the point we are using
        if mean < X_t:
            point = mean + dist_add
        else:
            point = mean - dist_add
        return point

    
    def get_use_two(self):
        if USE_TWO == True:
            return 2
        else:
            return 1


class data_info:
    def __init__(self, X, y, clf):
        '''
        Calculate the variables needed for the optimisation
         - R_ests, D, etc.
        '''
        data = {'X': X, 'y': y}
        # get projection
        self.proj_data = projection.from_clf(data, clf, supports=False)
        # split into classes
        self.xp1, self.xp2 = self.proj_data['X1'], self.proj_data['X2']
        # calc empirical means
        self.emp_xp1 = np.mean(self.xp1)
        self.emp_xp2 = np.mean(self.xp2)
        # distances from means
        self.d_1 = self.order_distances_from_mean(
            self.emp_xp1, self.xp1)
        self.d_2 = self.order_distances_from_mean(
            self.emp_xp2, self.xp2)
        # class Ns
        self.N1, self.N2 = len(self.xp1), len(self.xp2)
        # calc radius estimates
        self.R1_emp = radius.supremum(self.xp1, self.emp_xp1)
        self.R2_emp = radius.supremum(self.xp2, self.emp_xp2)
        # Empirical D
        self.D_emp = np.abs(self.emp_xp1 - self.emp_xp2)
        # stability numbers -- min error on empirical mean
        self.min_conc_1 = radius.error_upper_bound(self.R1_emp, self.N1, 0.9999999999999999)
        self.min_conc_2 = radius.error_upper_bound(self.R2_emp, self.N2, 0.9999999999999999)

    def order_distances_from_mean(self, mean, X):
        dists = np.sqrt(np.sum(np.square(X - mean), axis=1))
        return np.sort(dists)


    def __repr__(self):
        print_info = "Data Info contains the following attributes:\n"
        for key in self.__dict__:
            print_info += f"    {key}\n"
        return print_info

    def __call__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            raise AttributeError(f"{attr} not in data_info.\n {self}")
        