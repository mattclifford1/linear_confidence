'''
Moved to deltas/legacy/ecai2024/downsample.py (frozen: the published ECAI 2024 method).

This module is an alias of it: `deltas.model.downsample` *is* that module object, so old
imports, isinstance checks, pickles and monkeypatches keep working unchanged.
'''
import sys

from deltas.legacy.ecai2024 import downsample as _legacy

sys.modules[__name__] = _legacy
