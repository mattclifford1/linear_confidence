'''
Projected data as every modular deltas method sees it: two one-dimensional
samples, one per class, each knowing which side of the boundary it sits on.

This is the successor of the `data_info` dict (legacy base.py) and class
(legacy data_info.py) for the modular code. The legacy estimators keep theirs.
'''
import numpy as np


class ClassSample:
    '''
    one class's projected scores, sorted

        side = 'low'   the class sits below the boundary: its errors are the
                       points with z > b
        side = 'high'  the class sits above the boundary: its errors are the
                       points with z <= b
    '''

    def __init__(self, z, label, side):
        if side not in ('low', 'high'):
            raise ValueError("side must be 'low' or 'high'")
        self.z = np.sort(np.asarray(z, dtype=float).ravel())
        self.label = label
        self.side = side
        self.N = len(self.z)

    def wrong_side_count(self, b):
        '''
        number of training points on the wrong side of each boundary in b

        A high point sitting exactly on b is predicted low (the rule is
        z <= b -> low) but is *not* counted here. At the midpoint candidates
        the distinction never arises; at a chosen boundary it can, and this is
        the convention the overlap methods have always used, so it is kept.
        '''
        b = np.asarray(b, dtype=float)
        if self.side == 'low':
            return self.N - np.searchsorted(self.z, b, 'right')
        return np.searchsorted(self.z, b, 'left')

    def facing_distance(self, b):
        '''signed distance from the class mean towards the boundary'''
        b = np.asarray(b, dtype=float)
        return b - self.mean if self.side == 'low' else self.mean - b

    @property
    def mean(self):
        return float(np.mean(self.z))

    @property
    def std(self):
        '''sample standard deviation (ddof = 1); nan below two points'''
        return float(np.std(self.z, ddof=1)) if self.N > 1 else np.nan

    @property
    def radius(self):
        '''the published method's two-sided empirical radius max|z - mean|'''
        return float(np.max(np.abs(self.z - self.mean)))

    def __repr__(self):
        return (f'ClassSample(label={self.label}, side={self.side!r}, '
                f'N={self.N})')


class ProjectedData:
    '''
    both classes, oriented: the class with the smaller mean sits below the
    boundary (predicted when z <= b)
    '''

    def __init__(self, z, y):
        z = np.asarray(z, dtype=float).ravel()
        y = np.asarray(y).ravel()
        z0, z1 = z[y == 0], z[y == 1]
        if len(z0) == 0 or len(z1) == 0:
            raise ValueError('One class has no data points')
        # means of the unsorted arrays, exactly as the overlap methods always
        # computed them: the order of summation can matter at an exact tie
        if np.mean(z0) <= np.mean(z1):
            self.low = ClassSample(z0, 0, 'low')
            self.high = ClassSample(z1, 1, 'high')
        else:
            self.low = ClassSample(z1, 1, 'low')
            self.high = ClassSample(z0, 0, 'high')
        self.class_nums = [self.low.label, self.high.label]
        self.N = {0: len(z0), 1: len(z1)}

    def by_label(self, label):
        return self.low if self.low.label == label else self.high

    def sides(self):
        return (('low', self.low), ('high', self.high))

    def __repr__(self):
        return f'ProjectedData(low={self.low}, high={self.high})'
