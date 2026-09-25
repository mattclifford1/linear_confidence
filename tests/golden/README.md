# Golden tests for the legacy deltas code

These pin the **exact** outputs the legacy estimators produce today. Nothing may
change them during the modular refactor: the published method, the non-separable
variants, the overlap methods and the exploratory estimators are *moved*, not
rewritten.

| file | what |
|---|---|
| `fixtures/*.npz` | pre-computed 1-D projections (train/test scores, labels, threshold) — see `provenance.json` |
| `make_fixtures.py` | builds the fixtures; only rerun to add one |
| `cases.py` | every legacy estimator × fixture, run through its current import path; `--use-two {true,false}` prints JSON |
| `record.py` | writes `expected_use_two_{true,false}.json`, each from a fresh interpreter |
| `../test_golden_legacy.py` | recomputes both and compares exactly |

Fixtures: `gauss_wide` (the only one the published method solves without slacks),
`gauss_sep` (separable, yet infeasible without slacks), `gauss_overlap`,
`breast_cancer` and `pima` (the published pipeline's SVM-rbf projections, seed 0).

```bash
uv run pytest tests -m golden          # just these (~30 s)
uv run pytest tests -m "not golden"    # everything else
uv run python tests/golden/record.py   # re-record: only when a number is MEANT to move
```

Recorded outputs include known bugs that feed the published numbers
(`FINDINGS.md` §7.1) — e.g. `base_deltas` raising `TypeError` on an infeasible
problem (B11). They are pinned deliberately: fixing one is a separate decision,
made visible by re-recording.

`USE_TWO` is bound at import time, which is why each setting runs in its own
interpreter. `test_use_two_actually_changes_the_published_method` checks the
mechanism works — if the flag stopped taking effect both files would agree.
