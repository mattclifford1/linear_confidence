'''
Fingerprints that tell a cache its inputs changed: code (ignoring comments and
docstrings), settings, and data.
'''
import ast
import hashlib
import inspect
import json
import os
import textwrap
import types

import numpy as np


def digest(obj, n=12):
    return hashlib.sha1(json.dumps(stable(obj), sort_keys=True).encode()).hexdigest()[:n]


def arrays_hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f'{a.dtype}{a.shape}'.encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def source_hash(src):
    return hashlib.sha1(ast.dump(_strip_docstrings(ast.parse(src))).encode()).hexdigest()[:12]


def _constants(fn, skip):
    '''the upper-case module globals fn reads, by name'''
    names, stack = set(), [fn.__code__]
    while stack:
        co = stack.pop()
        names.update(n for n in co.co_names if n.isupper())
        stack += [c for c in co.co_consts if isinstance(c, types.CodeType)]
    g = fn.__globals__
    return {n: stable(g[n]) for n in sorted(names - set(skip)) if n in g}


def code_hash(*fns, skip=()):
    '''functions' source and the upper-case constants they read, except `skip`'''
    parts = []
    for fn in fns:
        fn = inspect.unwrap(fn)
        parts.append([fn.__qualname__,
                      source_hash(textwrap.dedent(inspect.getsource(fn))),
                      _constants(fn, skip)])
    return digest(parts)


def package_hash(root, skip=()):
    '''every .py file under root, outside the `skip` subfolders'''
    h = hashlib.sha1()
    for dirpath, dirs, files in sorted(os.walk(root)):
        dirs[:] = sorted(d for d in dirs if d not in skip and d != '__pycache__')
        for name in sorted(files):
            if name.endswith('.py'):
                path = os.path.join(dirpath, name)
                with open(path, encoding='utf-8') as fh:
                    h.update(os.path.relpath(path, root).encode())
                    h.update(source_hash(fh.read()).encode())
    return h.hexdigest()[:12]


def stable(obj):
    '''a JSON-able form of obj that is the same across processes'''
    if isinstance(obj, dict):
        return {str(k): stable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = [stable(v) for v in obj]
        return sorted(items, key=json.dumps) if isinstance(obj, (set, frozenset)) else items
    if isinstance(obj, np.ndarray):
        return arrays_hash(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, types.CodeType):
        return [obj.co_name, obj.co_code.hex(), stable(obj.co_consts),
                list(obj.co_names)]
    if isinstance(obj, types.FunctionType):
        try:                                     # source survives Python upgrades
            return ' '.join(inspect.getsource(obj).split())
        except (OSError, TypeError):
            return stable(obj.__code__)
    if isinstance(obj, type):
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return f'{type(obj).__module__}.{type(obj).__qualname__}'
