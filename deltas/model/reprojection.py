'''
scikit-learn style class to fit deltas
'''
import deltas.plotting.plots as plots
import deltas.classifiers.models as models
from deltas.model import base


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
    
    def get_suppport_inds(self):
        return self.projecter.support_
    
    def get_support_vectors(self):
        return self.projecter.support_vectors_


class reprojection_deltas(base.base_deltas):
    '''
    use a model (e.g SVM) to project current classisfier onto 1D projection
    This method doesn't have the requirement that the classifier needs a get_projection method to 1D,
    but it does need the classifier to project to a feature space (not implimented yet)
    '''

    def __init__(self, clf, projection_model=reprojectioner, *args, **kwargs):
        super().__init__(clf, *args, **kwargs)
        self.projection_model = projection_model()

    def fit(self, X, y, _plot=False, **kwargs):
        clf_copy = self.clf

        # fit model (SVM) to the projection
        self.clf = self.projection_model.fit_projection(clf=clf_copy, X=X, y=y)

        if _plot == True:
            plots.projections_from_data_clfs([clf_copy, self.clf], X, y)

        super().fit(X, y, _plot=_plot, **kwargs)
        return self
