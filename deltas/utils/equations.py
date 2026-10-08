'''
Moved to deltas/legacy/ecai2024/equations.py (frozen: the published ECAI 2024 method).

This module is an alias of it: `deltas.utils.equations` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.ecai2024 import equations as _legacy

sys.modules[__name__] = _legacy
