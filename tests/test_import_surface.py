'''
Every deltas import path that notebooks, scripts and experiments use today must
keep working.

The modular refactor moves the legacy estimators into deltas/legacy/ and leaves
import shims behind; this is what proves the shims are complete. The list was
built by grepping notebooks*/, dev/, experiments/, jonny/ and tests/ for
`deltas` imports (2026-09-25). Three imports found there were already broken
before the refactor and are deliberately absent: deltas.pipeline.run,
deltas.model_deltas and deltas.classifiers.large_margin_train. The torch-only
modules (MNIST/MIMIC nets) are skipped because torch is an optional extra.
'''
import importlib

import pytest

#: module -> names that callers import from it or reach through it
SURFACE = {
    'deltas': [],
    'deltas.model.base': ['base_deltas'],
    'deltas.model.downsample': ['downsample_deltas'],
    'deltas.model.non_sep': ['deltas'],
    'deltas.model.data_info': ['data_info'],
    'deltas.model.overlap': ['binomial_deltas', 'dkw_deltas',
                             'base_overlap_deltas', 'clopper_pearson_upper',
                             'dkw_upper'],
    'deltas.model.SSL': ['SSL_deltas'],
    'deltas.model.SVM_supports': ['SVM_supports_deltas'],
    'deltas.model.reprojection': ['reprojection_deltas', 'reprojectioner'],
    'deltas.optimisation.optimise_deltas': ['optimise'],
    'deltas.optimisation.optimise_contraint': ['get_init_deltas'],
    'deltas.utils.equations': ['loss', 'loss_one_delta', 'contraint_eq7',
                               'delta2_given_delta1_matt', 'J_derivative'],
    'deltas.utils.radius': ['R_upper_bound', 'error_upper_bound', 'supremum'],
    'deltas.utils.projection': ['from_clf', 'make_calcs'],
    'deltas.utils.data': ['split_classes'],
    'deltas.utils.cache': ['cached', 'info', 'clear'],
    'deltas.misc.use_two': ['USE_TWO', 'USE_GLOBAL_R', 'RANDOM_STATE'],
    'deltas.plotting.plots': ['plot_projection', 'deltas_projected_boundary',
                              'conc_projected_boundary'],
    'deltas.pipeline.data': ['get_real_dataset', 'get_data'],
    'deltas.pipeline.classifier': ['get_classifier'],
    'deltas.pipeline.evaluation': ['eval_test'],
    'deltas.pipeline.cached': ['get_data_and_classifiers',
                               'get_deltas_fit_data'],
    'deltas.pipeline.calibration': ['split_calibration', 'fit_calibrated'],
    'deltas.pipeline.pipeline_old': [],
    'deltas.classifiers.models': ['SVM', 'linear', 'NN'],
    'deltas.classifiers.frozen': ['FrozenProjection'],
    'deltas.classifiers.sibling': ['build', 'as_deltas_classifier'],
    'deltas.data.loaders.sibling': ['get_sibling_dataset'],
    'deltas.data.utils': ['shuffle_data', 'normaliser'],
    'deltas.costcla_local.models': ['Thresholding', 'BMR'],
}


@pytest.mark.parametrize('module', sorted(SURFACE))
def test_import_path_resolves(module):
    mod = importlib.import_module(module)
    missing = [name for name in SURFACE[module] if not hasattr(mod, name)]
    assert not missing, f'{module} lost {missing}'



#: old path -> where the frozen code now lives
MOVED = {
    'deltas.model.base': 'deltas.legacy.ecai2024.base',
    'deltas.model.downsample': 'deltas.legacy.ecai2024.downsample',
    'deltas.utils.equations': 'deltas.legacy.ecai2024.equations',
    'deltas.utils.radius': 'deltas.legacy.ecai2024.radius',
    'deltas.optimisation.optimise_deltas': 'deltas.legacy.ecai2024.optimise_deltas',
    'deltas.optimisation.optimise_contraint': 'deltas.legacy.ecai2024.optimise_contraint',
    'deltas.model.non_sep': 'deltas.legacy.non_separable.non_sep',
    'deltas.model.data_info': 'deltas.legacy.non_separable.data_info',
    'deltas.model.SSL': 'deltas.legacy.exploratory.SSL',
    'deltas.model.reprojection': 'deltas.legacy.exploratory.reprojection',
    'deltas.model.SVM_supports': 'deltas.legacy.exploratory.SVM_supports',
}


@pytest.mark.parametrize('old', sorted(MOVED))
def test_old_path_is_the_same_module_object(old):
    '''an alias, not a copy: classes, globals and monkeypatches are shared'''
    assert importlib.import_module(old) is importlib.import_module(MOVED[old])


def test_from_package_import_gives_the_legacy_module():
    from deltas.model import downsample, non_sep
    from deltas.legacy.ecai2024 import downsample as d2
    from deltas.legacy.non_separable import non_sep as n2
    assert downsample is d2 and non_sep is n2
    assert downsample.downsample_deltas.__module__ == \
        'deltas.legacy.ecai2024.downsample'
