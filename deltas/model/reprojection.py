'''
Moved to deltas/legacy/exploratory/reprojection.py (frozen: exploratory, used by neither paper).

This module is an alias of it: `deltas.model.reprojection` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.exploratory import reprojection as _legacy

sys.modules[__name__] = _legacy
