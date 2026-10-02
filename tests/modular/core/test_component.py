'''
The one base every component shares: describe() has to be JSON-friendly for results files.
'''
from deltas.bounds import ClopperPearson
from deltas.confidence import OptimisedDelta
from deltas.rules import Minimax
from deltas.search import Grid
from deltas.transforms import Logit


def test_describe_is_json_friendly():
    import json
    for c in (ClopperPearson(), OptimisedDelta(), Minimax(), Grid(), Logit()):
        json.dumps(c.describe())
