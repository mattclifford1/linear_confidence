# The calibration split

*What it solves, why it is needed, and what it costs.*

Implementation: `deltas/pipeline/calibration.py`.
Option: `deltas.pipeline.cached.get_data_and_classifiers(..., calibration=0.35)`.
Tests: `tests/test_calibration.py`.
Measurement: `experiments/validate_bounds.py`.

---

## 1. What deltas claims

Every deltas variant turns a statistic read off the projected training data into
a confidence statement about future data:

| variant | statistic read | claim made |
|---|---|---|
| published separable method (ECAI 2024) | empirical class support `R̂ᵢ` | class-`i` error ≤ something, w.p. `1 − δᵢ` |
| `model/overlap.py` (Clopper–Pearson, DKW) | miscount `mᵢ(b)` | `eᵢ(b) ≤ Uᵢ`, w.p. `1 − δᵢ` |

The overlap methods are the ones that make this checkable, because they *report*
the pair `(δᵢ, Uᵢ)` they certified. `certified_error()` returns it. You can
therefore compare `Uᵢ` against the error actually measured on held-out data and
ask whether the guarantee held.

It often does not.

## 2. The problem

The Clopper–Pearson bound rests on

```
mᵢ(b) ~ Binomial(Nᵢ, eᵢ(b))
```

which requires the projected points `z = ⟨φ(x), w⟩` to be i.i.d. draws from the
projected class-conditional law.

**They are not, when the classifier that defines the projection `w` was fitted
to those same points.** Training pushed them towards their own side of the
boundary. `mᵢ` therefore under-counts, `Uᵢ` comes out too small, and the
certificate is violated more often than `δᵢ` permits. The same argument applies
to `R̂ᵢ` in the published method: an empirical support computed on the
classifier's own training data is systematically too small.

This is **not a defect in the bound.** Clopper–Pearson is exact and DKW is
valid; the failure is in an assumption upstream of them.

### Measured, on the wide grid

`experiments/run_wide.py` + `analyse_wide.py`, **32 datasets × 7 model families
× 10 seeds**, 35,536 certificate observations:

| certificate computed on | coverage | nominal |
|---|---|---|
| the training data | **0.684** | 0.948 |
| a 35% held-out calibration split | 0.949 | 0.878 |

The certificate is wrong about **one time in three**. The shortfall tracks
classifier optimism monotonically, and the model ordering is essentially the
ordering of how hard each family overfits:

| model | naive | split |
|---|---|---|
| NearestClassMean | 0.891 | 0.941 |
| Logistic regression | 0.798 | 0.953 |
| MLP | 0.792 | 0.949 |
| LDA | 0.693 | 0.944 |
| SVM-rbf | 0.571 | 0.950 |
| RandomForest | 0.553 | 0.955 |
| GradientBoosting | **0.486** | 0.948 |

Every family is repaired to 0.94–0.96, from starting points ranging over 0.49
to 0.89 — the fix is not model-specific. `NearestClassMean` being least
affected is a nice consistency check: it barely overfits, and its
centroid-plus-radius geometry is exactly what the published derivation assumes.

See `FINDINGS.md` §11 for the full breakdown.

### Measured, earlier and smaller

`experiments/validate_bounds.py`, 4 datasets × 10 seeds × 2 classes,
Clopper–Pearson, minimax rule:

| certificate computed on | observed coverage | nominal |
|---|---|---|
| the training data | **0.81** | 0.96 |
| a 35% held-out calibration split | **1.00** | 0.90 |

And the shortfall is not noise — it tracks how much the classifier overfits:

| dataset | train err | test err | optimism | coverage |
|---|---|---|---|---|
| Breast Cancer | .054 | .067 | .013 | 0.90 |
| Heart Disease | .121 | .148 | .027 | 0.95 |
| Pima Diabetes | .177 | .298 | .121 | 0.75 |
| Hepatitis | .093 | .301 | **.208** | **0.65** |

An RBF SVM on 195 Hepatitis points overfits hard, and its certificate is wrong
a third of the time.

The obvious rival explanation — that `δ` is chosen by looking at the data, since
the method minimises the loss over a `δ` grid — was tested and ruled out.
Repeating with a *pre-specified* `δ ∈ {0.05, 0.10}` still gives 0.85 / 0.83.
It is the projection, not the `δ` selection.

## 3. The method

Sample splitting, exactly as in split-conformal prediction:

```
1.  partition the training data into a FIT part and a CALIBRATION part,
    stratified by class so both keep minority points
2.  train the classifier on the FIT part only
3.  compute the deltas certificate on the CALIBRATION part
```

Conditional on the fitted projection `w`, the calibration points were never
seen by the classifier, so they *are* i.i.d. draws from the projected law and
the binomial model holds exactly.

The guarantee then reads: **for this deployed classifier, the class-`i` error is
at most `Uᵢ` with probability at least `1 − δᵢ`** — conditional on the fitted
model, which is the statement a practitioner actually wants.

### Two distinct problems, two distinct fixes

It is worth being precise, because they are easy to conflate:

| problem | fix | where |
|---|---|---|
| the projection was fitted to the counted points | **calibration split** | `pipeline/calibration.py` |
| the boundary `b` is *chosen* using the same counts | **union bound over thresholds** (CP) / **DKW uniformity** | `model/overlap.py` |

The split alone is not enough: even on fresh calibration data, `b` is selected
by minimising over candidate thresholds, so `mᵢ(b*)` at the selected `b*` is not
a fixed-threshold binomial. That is what `_delta_correction` pays for — the
Clopper–Pearson path divides `δ` by `N + 1` (the number of distinct values `mᵢ`
can take), and DKW needs no correction because it is uniform in `b` by
construction. Both mechanisms are needed for the certificate to be true.

## 4. Why it cannot live inside the estimator

By the time `deltas_model.fit(X, y, clf=clf)` runs, `clf` is already trained.
Splitting `X` *inside* that call would compute the count on points the
classifier had already fitted to, which changes nothing and would be actively
misleading — it would look like a fix while leaving the bias in place.

The split has to happen **upstream of classifier training**, which is why it is
a pipeline option and not an estimator argument.

## 5. Using it

Through the cached pipeline:

```python
from deltas.pipeline import cached
from deltas.model import overlap

data_clf, clfs = cached.get_data_and_classifiers(
    'Hepatitis', 'SVM-rbf', seed=0, calibration=0.35)

X, y = cached.get_deltas_fit_data(data_clf)   # the calibration part
model = overlap.binomial_deltas(clfs['Baseline'], objective='minimax').fit(X, y)
model.certified_error()                       # now an honest certificate
```

`calibration=None` (the default) keeps the old behaviour. The untouched
training set stays at `data_clf['data_full']`; the classifiers are trained on
`data_clf['data']`, which is now the fit part.

Self-contained version, no pipeline:

```python
from deltas.pipeline import calibration

model, info = calibration.fit_calibrated(
    clf_factory=lambda: models.SVM(kernel='rbf'),
    deltas_factory=lambda clf: overlap.binomial_deltas(clf, objective='minimax'),
    X=X, y=y, cal_frac=0.35, seed=0)
```

Note the classifier is passed as a **factory**, not an instance — the whole
point is that it gets trained on the fit part rather than handed over
pre-trained.

The cache key includes `calibration`, so calibrated and uncalibrated runs never
share an entry. It is only added to the key when it is set, so existing
uncalibrated cache entries keep their hashes.

## 6. What it costs

Both costs are real and neither is hidden:

- **A looser certificate.** Mean `Uᵢ` roughly doubles (.32 → .52 at
  `cal_frac=0.35`), because the count now comes from a third of the data and
  `Uᵢ` shrinks like `1/N_cal`.
- **A worse classifier.** It is trained on 65% of the data.

That is the price of the certificate being true rather than approximately true.
`cal_frac` trades the two off and has not been tuned; 0.35 is what the
measurements above used.

There is also a hard floor. `split_calibration` **raises**
`CalibrationSplitError` rather than return a split that leaves fewer than 2
points of a class in either part. A certificate computed from one minority
point is worthless, and returning it silently would be worse than failing. On
the scarcest datasets this means the split is simply unavailable.

## 7. Scope

**This applies to the published ECAI method too.** Its `R̂ᵢ` is the empirical
support of the classifier's own training data, so its `1 − δᵢ` statements carry
the same optimism. It has never surfaced because that method never reports the
quantity it certifies — there is nothing to check it against. The overlap
formulation did not introduce this problem; it made an existing one visible.

## 8. What it does *not* fix

- **It does not make `Uᵢ` tight.** A distribution-free bound from `N_cal`
  minority points cannot certify an error below roughly `ln(1/δ)/N_cal`
  whatever the boundary. Splitting makes that floor *higher*, not lower.
- **It does not fix a badly chosen `b`.** Coverage is about the certificate
  being honest, not about the boundary being good. The two are separate.
- **It does not transfer across distribution shift.** The guarantee is
  conditional on the fitted classifier and on calibration and test data coming
  from the same law.
- **It says nothing about the classifier's own generalisation.** It certifies
  the error of the *thresholded projection*, given the projection.
