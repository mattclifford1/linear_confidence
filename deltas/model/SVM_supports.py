'''
Moved to deltas/legacy/exploratory/SVM_supports.py (frozen: exploratory, used by neither paper).

This module is an alias of it: `deltas.model.SVM_supports` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.exploratory import SVM_supports as _legacy

sys.modules[__name__] = _legacy
