'''
Moved to deltas/legacy/exploratory/SSL.py (frozen: exploratory, used by neither paper).

This module is an alias of it: `deltas.model.SSL` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.exploratory import SSL as _legacy

sys.modules[__name__] = _legacy
