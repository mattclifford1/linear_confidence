# `deltas/model` — old import paths

This folder now only keeps old imports working:

- `base.py`, `downsample.py`, `non_sep.py`, `data_info.py`, `SSL.py`,
  `reprojection.py`, `SVM_supports.py` are **aliases** of the frozen modules in
  `deltas/legacy/` (the same module objects). See `deltas/legacy/README.md`.
- `overlap.py` is a **shim** over the modular `DeltasEstimator`
  (`deltas/core/`). It keeps `binomial_deltas` / `dkw_deltas`, their signatures
  and their exact numbers (checked against `deltas/legacy/overlap/`). As
  sklearn estimators they now return real `get_params()` (was `{}`), reject
  unknown names in `set_params()`, and no longer have the private
  `_sparse_loss_table` helper (see `FINDINGS.md` §15).

New work: `deltas/core/` and the slot folders, or a named method from
`deltas/methods/`. See `deltas/README.md`.
