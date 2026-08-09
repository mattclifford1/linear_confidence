'''
disk cache for expensive experiment steps (dataset loading, classifier training)

Everything is keyed by a hash of a config dict plus the versions of the
libraries that can change the result. A stale cache therefore cannot silently
poison results - change anything relevant and you get a new key.
'''
import hashlib
import json
import os

import joblib
import numpy as np
import sklearn


CACHE_DIR = os.environ.get(
    'DELTAS_CACHE',
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'cache'))

# bump this by hand if the meaning of a cached artefact changes but the
# config dict does not (e.g. a bug fix inside get_classifier)
CACHE_VERSION = 1


def _env_fingerprint():
    '''library versions that can change a cached artefact'''
    return {'sklearn': sklearn.__version__,
            'numpy': np.__version__,
            'cache_version': CACHE_VERSION}


def make_key(kind, config):
    '''
    stable hash of (kind, config, env). config must be json serialisable
    '''
    payload = {'kind': kind,
               'config': config,
               'env': _env_fingerprint()}
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
    return f'{kind}-{digest}'


def _paths(key):
    folder = os.path.join(CACHE_DIR, key.split('-')[0])
    return folder, os.path.join(folder, f'{key}.joblib')


def load(key):
    '''return the cached object, or None if not present/unreadable'''
    _, path = _paths(key)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        # a corrupt/partial entry should never take down a run
        return None


def _picklable(value):
    try:
        joblib.dump(value, os.devnull)
        return True
    except Exception:
        return False


def sanitise(obj, _depth=0, _max_depth=3):
    '''
    make a nested dict safe to pickle

    Some loaders stash live third-party objects in the data dict - e.g.
    heart_disease.get_HD puts a ucimlrepo metadata object into
    'feature_names' and 'description' - which joblib cannot serialise. Those
    fields are descriptive only, so replace anything unpicklable with its
    string form rather than failing the whole cache write.
    '''
    if isinstance(obj, dict) and _depth < _max_depth:
        return {k: sanitise(v, _depth + 1, _max_depth) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray, list, tuple, str, int, float, bool,
                        type(None))):
        return obj
    return obj if _picklable(obj) else str(obj)


def save(key, obj):
    folder, path = _paths(key)
    os.makedirs(folder, exist_ok=True)
    # write to a temp file then rename so an interrupted run cannot leave a
    # half-written entry that looks valid
    tmp = path + '.tmp'
    joblib.dump(obj, tmp)
    os.replace(tmp, path)
    return path


def cached(kind, config, compute, use_cache=True, _print=False):
    '''
    return compute() but memoised on disk against (kind, config, env)
        kind:     short string namespace, e.g. 'dataset' or 'classifiers'
        config:   json serialisable dict identifying the artefact
        compute:  zero-arg callable producing the artefact
        use_cache: set False to bypass entirely (still writes the result)
    '''
    key = make_key(kind, config)
    if use_cache == True:
        obj = load(key)
        if obj is not None:
            if _print == True:
                print(f'  cache hit  {key}')
            return obj
    if _print == True:
        print(f'  cache miss {key} - computing')
    obj = compute()
    save(key, obj)
    return obj


def clear(kind=None):
    '''delete cached entries (all, or just one namespace)'''
    import shutil
    target = CACHE_DIR if kind is None else os.path.join(CACHE_DIR, kind)
    if os.path.exists(target):
        shutil.rmtree(target)
    return target


def info():
    '''summarise what is currently cached'''
    out = {}
    if not os.path.exists(CACHE_DIR):
        return out
    for kind in sorted(os.listdir(CACHE_DIR)):
        folder = os.path.join(CACHE_DIR, kind)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith('.joblib')]
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in files)
        out[kind] = {'entries': len(files), 'MB': round(size / 1e6, 1)}
    return out
