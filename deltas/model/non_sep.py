'''
Moved to deltas/legacy/non_separable/non_sep.py (frozen: the non-separable draft, Dec 2024).

This module is an alias of it: `deltas.model.non_sep` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.non_separable import non_sep as _legacy

sys.modules[__name__] = _legacy
