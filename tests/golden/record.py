'''
Record the golden expectations for the legacy deltas code.

    uv run python tests/golden/record.py

Writes expected_use_two_true.json (the HEAD setting) and
expected_use_two_false.json (the setting the ECAI 2024 tables were produced
with). Each is computed in a fresh interpreter, because USE_TWO is read at
import time.

Only re-record when a change is *meant* to move a legacy number, and say so in
the commit message: the whole point of these files is that the modular refactor
must not.
'''
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, 'cases.py')


def compute(flag):
    '''run every golden case in a fresh interpreter with USE_TWO = flag'''
    out = subprocess.run([sys.executable, CASES, '--use-two', flag],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def expected_path(flag):
    return os.path.join(HERE, f'expected_use_two_{flag}.json')


def main():
    import numpy
    import scipy
    import sklearn
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    for flag in ('true', 'false'):
        results = compute(flag)
        results['_meta'] = {
            'recorded': datetime.now().isoformat(timespec='seconds'),
            'git_commit': commit,
            'use_two': flag,
            'python': sys.version.split()[0],
            'numpy': numpy.__version__,
            'scipy': scipy.__version__,
            'sklearn': sklearn.__version__,
        }
        with open(expected_path(flag), 'w') as f:
            json.dump(results, f, indent=1, sort_keys=True)
        print(f'wrote {expected_path(flag)} ({len(results) - 1} entries)')


if __name__ == '__main__':
    main()
