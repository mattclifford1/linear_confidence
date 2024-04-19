'''
scikit-learn style class to fit deltas
'''
import numpy as np

from deltas.model import downsample, reprojection, base
import deltas.classifiers.models as models


# class SVM_supports_deltas(base.base_deltas):
class SVM_supports_deltas(downsample.downsample_deltas):
    '''
    use a model (e.g SVM) to project current classisfier onto 1D projection
    This method doesn't have the requirement that the classifier needs a get_projection method to 1D,
    but it does need the classifier to project to a feature space (not implimented yet)
    '''

    def __init__(self, clf, 
                 kernel='linear',
                 *args, 
                 **kwargs):
        super().__init__(clf, *args, **kwargs)
        self.projection_model = reprojection.reprojectioner(
            projecter=models.SVM(kernel=kernel))

    def fit(self, X, y, method='supports-prop-update_mean-margin_only', _plot=False, costs=(1, 1), **kwargs):
        clf_copy = self.clf
        data_info = self.get_data_info(X, y, clf_copy, _print=False, costs=costs)
        if _plot == True:
            print('Original Classifier')
            self._plot_data(data_info, self.clf)

        # fit SVM to the projection
        self.clf = self.projection_model.fit_projection(clf=clf_copy, X=X, y=y)
        data_info = self.get_data_info(X, y, self.clf, _print=False, costs=costs)
        if _plot == True:
            print('Reprojected Classifier')
            self._plot_data(data_info, self.clf)

        # remove any supports from the dataset
        remove_inds = self.clf.get_suppport_inds()
        X = np.delete(X, remove_inds, axis=0)
        y = np.delete(y, remove_inds, axis=0)

        data_info = self.get_data_info(X, y, self.clf, _print=False, costs=costs)
        if _plot == True:
            print('Supports from SVM removed')
            self._plot_data(data_info, self.clf)

        print('Fitting deltas using slacks if required')
        super().fit(X, y, method=method, _plot=_plot, **kwargs)
        return self

