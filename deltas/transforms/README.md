# `deltas/transforms` — monotone maps of the score

A threshold rule is invariant to any strictly increasing map g:
{z > b} = {g(z) > g(b)}. So the choice costs nothing for the classifier, but it
matters to any bound that assumes a shape (a Gaussian after a logit, say).

| registry name | what | fitted? |
|---|---|---|
| `identity` | z | no |
| `logit` | log(p / (1 − p)), clipped | no |
| `standardise` | (z − mean) / sd | yes |
| `yeo_johnson` | a Yeo–Johnson power map, then standardised | yes |

A fitted transform must be fitted **before** the deltas fit, on the
classifier's fit split: choosing it on the calibration data voids the
certificate. The estimator refuses an unfitted one. Boundaries are reported in
the classifier's own units (`get_bias()` uses `inverse`).
