'''
ClassSample and ProjectedData: wrong-side counts, orientation by mean, statistics.
'''
import numpy as np
import pytest

from deltas.core import ClassSample, ProjectedData


def test_wrong_side_counts_follow_the_prediction_rule():
    low = ClassSample([0.0, 1.0, 2.0], 0, 'low')
    high = ClassSample([1.0, 2.0, 3.0], 1, 'high')
    b = np.array([-1.0, 1.0, 1.5, 5.0])
    # low is wrong when z > b
    assert list(low.wrong_side_count(b)) == [3, 1, 1, 0]
    # high is counted wrong when z < b (a point exactly on b is not counted)
    assert list(high.wrong_side_count(b)) == [0, 0, 1, 3]


def test_orientation_by_mean_and_empty_class():
    d = ProjectedData(np.r_[5.0, 6.0, 0.0, 1.0], np.array([0, 0, 1, 1]))
    assert d.class_nums == [1, 0] and d.low.label == 1
    assert d.by_label(0) is d.high
    with pytest.raises(ValueError, match='no data'):
        ProjectedData(np.r_[1.0, 2.0], np.array([0, 0]))


def test_sample_statistics():
    s = ClassSample([1.0, 2.0, 3.0, 10.0], 0, 'high')
    assert s.mean == 4.0
    assert s.radius == 6.0
    assert s.facing_distance(0.0) == 4.0     # high: distance is mean - b
    assert np.isnan(ClassSample([1.0], 0, 'low').std)
