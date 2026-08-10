# Findings — state of the `deltas` research code

Written 2026-08-07 after a full read of the repo, the published ECAI paper, the
non-separable draft and the review notes. This is the "where were we" document.

- Published paper: `../../Repos/Overleaf/deltas/ecai-2024-deltas-arxiv-3-sup-mat/m598.tex`
- Non-separable draft: `../../Repos/Overleaf/deltas/deltas-non-separable/main.tex`
- Review notes on the published paper: `../../Repos/Overleaf/deltas/claude-improvement-notes.md`

---

## 1. Timeline of the work

| When | What |
|---|---|
| Nov 2023 – Apr 2024 | Core method built. `base_deltas`, the eq.6/7/8 optimisation, dataset loaders. `f290601 final for ecai` (26 Apr 2024). |
| May – Jun 2024 | Side quests: MNIST + torch nets, large-margin loss (Elsayed et al.), MIMIC-III/IV loaders and MLP. Most of this did **not** make the paper — it lives in `dev/` and `deltas/classifiers/`. |
| Jul – Aug 2024 | Camera-ready experiments. Binary slacks (`downsample_deltas`), then continuous slacks (Appendix B). `results/`, `results-continuous/`. |
| Oct 2024 | Presentation material (`notebooks-ECAI/presentation/`, `bias.gif`). |
| Nov 2024 | `USE_TWO` ablation (`results-two/`, `results-two2/`) — re-running everything with/without the factor of 2 in the concentration bound. Then the non-separable work starts (`9d0d5e2 start fopr non sep`). |
| Nov – Dec 2024 | Non-separable: new generalisation error from the *k*-th furthest point, bias-sweep optimiser, four loss variants (min / max / mean / furthest). Results tables into the draft. **Last commit 13 Dec 2024.** |

## 2. Where the method stands

### 2.1 The published (separable + slacks) method — `deltas/model/downsample.py`

Working, reproducible, and the numbers in the paper come out of
`notebooks-ECAI/run_all.py` → `results-locked/`.

Mechanism: project data to 1-D with `clf.get_projection`; per class compute the
empirical mean, the empirical support `R̄ᵢ` and `Nᵢ`; require the two inflated
supports to exactly meet, `R̂₁ + R̂₂ = D̂` (`equations.contraint_eq7`); solve for
`δ₂` given `δ₁` (`delta2_given_delta1_matt`); grid-search `δ₁` against
`L = Σ (1−δᵢ)/(Nᵢ+1) + δᵢ` (`equations.loss` / `class_cost`); set the boundary
where the two half-spaces meet.

When the projected classes overlap the constraint is infeasible, so
`downsample_deltas` peels off support points (class-proportionally, optionally
re-computing the means) and retries, penalising the loss by
`α · (removed / N)` per class. That is the slack mechanism.

### 2.2 The non-separable follow-up — `deltas/model/non_sep.py`

This is the interesting, unfinished work, and it is *already* most of the way to
the "handle overlap from the ground up" goal.

The idea: instead of generalising from the furthest training point, generalise
from the **k-th furthest**. The exchangeability argument in the paper's
Appendix A generalises immediately —

  P(test point lands beyond the k-th furthest of N training points) = k/(N+1)

so overlap is expressed by *choosing an inner order statistic*, not by deleting
points. The mean-vs-expectation uncertainty (`min_conc_i = 2 R̄ᵢ/√Nᵢ`) stays
constant regardless of which order statistic you pick — which is exactly why
this is cleaner than slacks, where dropping points inflated the uncertainty.

Algorithm (`non_sep.deltas.optimise`): sweep candidate biases on a linspace
between the two empirical means (clipped by `min_conc`); for each bias and each
class, compute the achievable `(k, δ)` pairs; combine them into a per-class
loss; aggregate across `k` by `min` / `max` / `mean`, or use only the furthest
point (`only_furtherest_k=True`, which recovers the ECAI behaviour); take the
`argmin` over the bias sweep.

**Empirically** (from the draft's tables and `results-non-sep/`):
`min` and `furthest` are the good variants and are roughly tied with the ECAI
slacks method; `max` is bad (e.g. Breast Cancer 0.78 accuracy vs 0.95); `mean`
is mediocre. On MIMIC the `furthest` variant is the best G-Mean/F1 in the table.

**So the honest current status is: the new formulation matches but does not beat
the slacks method.** That is the gap to close before this is a paper.

---

## 3. The single most important theory issue

`non_sep._single_class_loss` currently is

```python
# loss = error*(1-delta) + delta     # <- commented out
# loss = error*(1-delta)             # <- commented out
loss = error*(delta)                 # <- in use
```

i.e. `L = Σᵢ δᵢ · kᵢ/(Nᵢ+1)`, which is what §3.1 of the draft writes down while
admitting "further thought and analysis is needed on how best to come up with a
loss/score".

This is a **product of a failure probability and an error bound**, and it is not
an upper bound on anything. The published loss, by contrast, *is* one: with
probability `1−δ` the error is at most `1/(N+1)`, and otherwise at most 1, so
`(1−δ)/(N+1) + δ` upper-bounds the expected error. The k-th-order-statistic
version of that same argument gives

  **Lᵢ(b) = (1 − δᵢ) · kᵢ(b)/(Nᵢ+1) + δᵢ**

which is the first commented-out line. This is a drop-in change, keeps every
guarantee of the published paper, and makes the non-separable method a strict
generalisation of it (`k=1` recovers ECAI exactly). It should be the default,
with `error*delta` kept as a documented ablation.

Worth also noting: the two forms have *opposite* degenerate behaviour.
`error*delta` is minimised by driving `δ → 0` (maximal confidence, ignore the
error term); `(1−δ)·error + δ` is dominated by `δ` and drives towards the
`δ → 1` end of the sweep. Neither is obviously right without the ablation, so
run it — this is a one-line change and a one-figure result.

---

## 4. Building overlap in from the ground up (the stated next direction)

Ranked, most-recommended first. All three keep the post-hoc, bias-only framing.

### A. Binomial / Clopper–Pearson tail bound — *contains the current theory as a special case*

For a fixed boundary `b`, the number `mᵢ(b)` of class-`i` training points on the
wrong side is `Binomial(Nᵢ, eᵢ(b))` where `eᵢ(b)` is the true class-conditional
error. The exact one-sided Clopper–Pearson upper bound gives, with probability
at least `1 − δᵢ`,

  eᵢ(b) ≤ BetaInv(1 − δᵢ ; mᵢ + 1, Nᵢ − mᵢ).

Properties that make this the right move:

- **No separability assumption anywhere.** `mᵢ > 0` is normal, not a pathology.
  The constraint that makes the current method infeasible under overlap simply
  does not exist — `b` ranges freely over ℝ.
- **`mᵢ = 0` reduces to ≈ `ln(1/δᵢ)/Nᵢ`**, the same order as the existing
  `1/(Nᵢ+1)`. So the published separable result is the `m = 0` corner of this,
  which is a genuinely nice unification to write up.
- **Class asymmetry is automatic** — the bound is much looser for the minority
  class at equal `m`, which is precisely the "extra space for the minority
  class" behaviour the paper argues for, now with a certificate instead of a
  heuristic.
- Sweeping `b` needs a union bound, but `mᵢ(b)` only takes `Nᵢ + 1` values, so
  a union over the distinct thresholds costs `log(Nᵢ + 1)` — cheap and exact.
- **`R` disappears entirely.** See §5 for why that matters.

Objective becomes an honest cost-sensitive risk bound:
`L(b) = Σᵢ cᵢ · UB(mᵢ(b), Nᵢ, δᵢ)` with `δᵢ` traded off as now.
(`data_info['c1'] / ['c2']` already exist for the costs and are always `(1,1)`
in every experiment — this would finally give them a use.)

### B. DKW / empirical-CDF bound — the simplest uniform-in-`b` version

The projected space is one-dimensional, so the empirical CDF is available and
Dvoretzky–Kiefer–Wolfowitz gives, with probability ≥ `1 − δᵢ`,
`sup_t |F̂ᵢ(t) − Fᵢ(t)| ≤ √(ln(2/δᵢ)/(2Nᵢ))`, **uniformly over all `b` at once** —
so no union bound and no re-derivation per threshold. Slightly looser than A,
but two lines of code and it is uniform, which makes the "sweep `b`" step
rigorous rather than ad hoc.

### C. Keep the order-statistic route but fix the loss

This is §3 above: the exchangeability bound `k/(N+1)` is already a
distribution-free tolerance-region result (Wilks 1941; the same argument
underpins conformal prediction), and it is *exact*, not just an upper bound.
It is the twin of A — A conditions on the threshold and bounds the error, C
conditions on the error rank and gives an exact probability. Both are defensible
"from the ground up" stories; A is easier to combine across classes and to
present as a risk bound.

### What all three share, and why they beat slacks

The slack mechanism answers "which points do I delete so my separability
assumption becomes true again". A/B/C answer "what is the error at this
boundary, given that points are on both sides" — which is the question. The
deletion rule (drop supports proportionally to the imbalance ratio) has no
justification in the paper and the continuous-slack variant produced an
essentially identical table (Appendix B vs Table 3), which is itself evidence
that the slack abstraction is not carrying any signal.

### Experiments that would sell it

1. **Overlap sweep.** Synthetic 2-Gaussian, vary `‖μ‖` from separable to heavy
   overlap, plot G-Mean vs overlap for {baseline, thresholding, BMR, ECAI
   slacks, new method}. The paper has one operating point; the failure of the
   slack method as overlap grows is the motivating figure and currently does
   not exist. Cheap to run (`data.get_data(m1=[-m,-m], m2=[m,m], ...)`).
2. **Feasibility rate.** Fraction of seeds for which the slacks method finds any
   solution vs the new method (which should be 100% by construction). This
   directly quantifies the limitation admitted in §6.1 of the paper — and it is
   currently *hidden* by the experiment runner (see §6.1 below).
3. **Bound vs actual.** Plot the certified error bound against measured test
   error per class per dataset. The bounds are never evaluated in the published
   paper. This is the single biggest gap the review notes identify.
4. **Imbalance sweep** at fixed overlap, `ρ = N₁/N₂ ∈ {1, 2, 5, 10, 50, 100}`.
5. **Ablation of the loss form** (§3): `(1−δ)e + δ` vs `δe` vs `(1−δ)e`.

---

## 5. Theory gaps in the published method worth knowing about

These are things the code reveals that the paper does not say.

1. **The `R` in the concentration bound is not the `R` in the theorem.**
   Theorem 2 (Shawe-Taylor & Cristianini) needs `R = sup‖φ(x)‖`, the radius of a
   ball in *feature* space containing the distribution's support. The code
   (`radius.R_upper_bound` with `USE_GLOBAL_R = False`, which is the setting all
   published results use) substitutes `R̄ᵢ`, the *empirical* class support in the
   *projected* space. These are different objects and the substitution is not
   justified in the paper. `USE_GLOBAL_R = True` uses `sup‖proj(x)‖` over all
   data, which is closer to the theorem but still projected-space. The comment
   in `deltas/misc/use_two.py` reads "`R` definition be class based of all data
   — matt thinks no", so this was a considered choice, but it is undocumented.
   **Approaches A and B in §4 need no `R` at all**, which removes the gap.

2. **`USE_TWO`: the factor of 2 is required, and the paper's Eq. 5 is missing
   it.** See §5A below — this deserves its own treatment.

3. **The empirical supremum `R̄ᵢ` has zero breakdown point.** One mislabelled or
   outlying point moves the boundary. Any quantile/order-statistic formulation
   (§4) fixes this for free.

4. **Eq. 5 is a fixed-point inequality used as a plug-in.** `R̂ᵢ ≤ R̄ᵢ + R̄ᵢ/√Nᵢ(…)`
   bounds `R̂ᵢ` in terms of `R̄ᵢ` while `R̄ᵢ` also appears on the right. Solving it
   properly tightens the bound.

5. **The boundary is a midpoint, not a meeting point.** `base._make_boundary`
   returns `(upper_min_class + lower_max_class)/2`. When the constraint is
   satisfied exactly these coincide, so this is a no-op in the solvable case —
   but it silently papers over near-misses.

---

## 5A. `USE_TWO` — should the factor of 2 be in the equation?

Short answer: **yes for the formulation as it stands (so HEAD's `True` is the
correct setting and the published `False` was not), but the honest fix is to
reformulate so the term disappears entirely.**

### The derivation

Write `ε(δ) = (R/√N)(2 + √(2 ln(1/δ)))` — one application of Theorem 2 to the
centroid deviation, bounding `‖φ̄_S − φ_S‖ ≤ ε(δ)` with probability `≥ 1 − δ`.
Let `d̄ₙ = |zₙ − φ̄_S|` (computable) and `dₙ = |zₙ − φ_S|` (what the
exchangeability argument is about).

We want to place the boundary at distance `t` from the **empirical** mean such
that a test point exceeding it is unlikely. Condition on the good event
`A = {‖φ̄_S − φ_S‖ ≤ ε}`:

- **Training points, empirical → true:** `dₙ ≤ d̄ₙ + ε ≤ R̄̂ + ε`, so
  `maxₙ dₙ ≤ R̄̂ + ε`.
- **Test point, true → empirical:** if `d̄_{N+1} > t` then `d_{N+1} > t − ε`.

Chaining these, `{d̄_{N+1} > t} ∩ A ⊆ {d_{N+1} > maxₙ dₙ}` **provided
`t − ε ≥ R̄̂ + ε`, i.e. `t ≥ R̄̂ + 2ε`**. Then exchangeability gives

  `P(d̄_{N+1} > t) ≤ δ + 1/(N+1)`,

which is exactly the loss the paper minimises.

So the 2 is not decoration: it is **one `ε` to lift the training maximum from
empirical into true coordinates, and a second to drop the test point back from
true into empirical coordinates**. Both are needed because `R̄̂` and `t` are both
expressed relative to the empirical mean.

Note also that both uses are covered by the *same* event `A`, so there is no
union bound to pay — `2ε(δ)`, not `2ε(δ/2)`. The code is right about that.

### The paper contradicts itself

- Eq. 4 (`eq:test_point_bound`) **has** the 2: `… + 2R/√N (2 + √(2 ln 1/δ))`,
  and the surrounding text spells out the reason ("upper-bounding the distances
  from the points in the training set and lower-bounding the distance from a
  test point").
- Eq. 5 (`R_hat_upper_bound`) **drops it**: `R̂ᵢ ≤ R̄̂ᵢ + (R̄̂ᵢ/√Nᵢ)(2 + √(2 ln 1/δᵢ))`.

`USE_TWO` is precisely the switch between these two readings, and the published
tables were generated from the Eq. 5 version. The comment in
`deltas/misc/use_two.py` — *"use original 2 in Dario's equations - matt thinks
no"* — records the disagreement. By the derivation above, Dario was right and
Eq. 5 is the typo.

### What this costs the published paper

The consequence is mild but should be stated: with `ε` instead of `2ε` the
boundary is placed *less* conservatively than the theory licenses, so the
claimed guarantee does not strictly hold at the stated `δᵢ` — the stated
confidence is optimistic. It does not invalidate the empirical conclusions; if
anything `USE_TWO=True` scores slightly better on several datasets (Breast
Cancer .944 → .951). The clean fix for a journal version is to correct Eq. 5,
re-run, and note that the results improve.

The flag is applied **consistently** everywhere it appears
(`radius.error_upper_bound`, `data_info.min_conc_i = 2R̄ᵢ/√Nᵢ`,
`equations.delta2_given_delta1_matt`, `equations.eq7_matt`,
`non_sep.deltas_from_bias`), so there is no partial-application bug — it is one
coherent choice, just the wrong one in the published run.

### …but the term should not exist at all

The `2ε` is provably loose, and the reason points straight at the fix.

In the projected space let `Δ = φ̄_S − φ_S` be the (scalar) mean shift. For
points **on one side** of the mean, `dₙ = d̄ₙ + Δ` for *every* point including
the test point — the same constant applied to all of them. So the comparison
`d_{N+1} > maxₙ dₙ` is **exactly equivalent** to `d̄_{N+1} > maxₙ d̄ₙ`: the shift
cancels identically and `ε` is not needed at all.

`ε` only enters because `R̄̂ᵢ = supₓ |z − φ̄_S|` is a **two-sided** quantity — a
ball — so the maximum can be attained on either side, and the shift no longer
cancels. That two-sidedness is inherited from a feature-space theorem about
balls, applied to a problem that is one-dimensional and one-sided: a threshold
classifier only ever cares about the tail of class 1 above `b` and the tail of
class 2 below it.

Drop the ball framing and order the raw projections `z` directly. Then for the
N+1 exchangeable points of class `i`,

  `P(z_{N+1} > all N training points) = 1/(Nᵢ+1)` exactly,

and more generally `k/(Nᵢ+1)` for the k-th largest — **distribution-free, exact,
no mean estimate, no `R`, no concentration inequality, and no factor of 2 to
argue about.**

This is the same conclusion §4 reaches from the other direction, and it makes
`USE_TWO` a non-question rather than a judgement call. It is also the strongest
argument for the reformulation: the entire `R/√N` apparatus — the part of the
method hardest to justify (see §5.1) and the part that makes the constraint
infeasible under overlap — is an artifact of measuring distances from an
estimated mean in a problem that never needed a mean.

### Recommendation

1. Set `USE_TWO = True` and leave it there for any further work on the current
   formulation; record it in results files (§6.3).
2. In the journal version, correct Eq. 5 to match Eq. 4 and re-run.
3. In the new (overlap-native) formulation, remove the term entirely by working
   with one-sided order statistics on the raw projection — then neither `ε` nor
   `R` nor the factor of 2 appears anywhere.

---

## 6. Reproducibility problems (blockers for "run all the experiments again")

### 6.0 ⚠️ HEAD does not reproduce the published paper — `USE_TWO` was flipped

**The ECAI results were produced with `USE_TWO = False`. HEAD has
`USE_TWO = True`.** The flag was flipped in commit `583e663` ("results from use
2", 12 Nov 2024) as part of starting the non-separable work, and never flipped
back. Nothing in any results file records which setting produced it.

Verified on Breast Cancer, SVM-rbf, seeds 0–9, 2026-08-07:

| config | Baseline (Acc/G-Mean/F1) | Our Method |
|---|---|---|
| HEAD (`USE_TWO=True`) | .912 / .912 / .908 | .951 / .950 / .954 |
| `USE_TWO=False` | **.917 / .917 / .914** | **.945 / .943 / .947** |
| published Table 3 | .917 / .917 / .913 | .943 / .942 / .946 |

The `USE_TWO=False` rerun reproduces `results/` exactly once the truncation bug
(§6.4) is accounted for: .945 truncates to .944, which is what `results/` says.
Published Table 3's .943 comes from a slightly later run whose per-dataset files
were overwritten — see the note below.

So the mapping is:

| Results dir | `USE_TWO` | Corresponds to |
|---|---|---|
| `results-locked/`, `results/`, `results-two/` | False | published Table 3 |
| `results-continuous/` | False | published Table 4 |
| `results-two2/`, `notebooks-non-sep/results-non-sep/` | **True** | the non-separable draft |

This also means **the non-separable draft's "Slacks Deltas" baseline row is not
the published method** — it is the published method with the other `USE_TWO`
setting (draft Breast Cancer slacks = .951/.949/.953 = `results-two2` exactly).
Any comparison in that draft between "Slacks Deltas" and the ECAI paper's
numbers is apples-to-oranges. Fix before the draft goes anywhere.

Also note: the exact per-dataset files behind published Table 3 no longer exist
on disk. Table 3's LaTeX comments are timestamped 26/08/2024 17:55–17:56 (SVM
datasets) and 21/08/2024 16:24:35 (MIMIC); `results/` was overwritten by a
later run at 17:36/18:11 the same day. `results-locked/` and `results/` agree
with Table 3 on Heart Disease and MIMIC but differ by one unit in the last digit
on Pima (.673 vs .672) and Breast Cancer (.944 vs .943).

To reproduce the paper: patch the flag **before** importing anything else from
`deltas`, because the modules bind it by value at import time:

```python
import deltas.misc.use_two as ut
ut.USE_TWO = False          # must come before the deltas.* imports below
from deltas.pipeline import data, classifier, evaluation
```

### 6.1 The seed loop is selection-biased and unrecorded — **fix this first**

`run_all.py` / `run_all_non_sep.py` do:

```python
for i in generator():                  # seeds 0, 1, 2, ... forever
    ...
    if deltas_model.is_fit == True:    # non_sep requires ALL FOUR variants to fit
        dfs.append(scores_df)
    if len(dfs) == len_required: break # stop at 10 successes
```

Consequences:

- Seeds where the method finds no solution are **dropped for every method**,
  including the baselines. So the reported baseline/SMOTE/BMR numbers are
  conditioned on "deltas succeeded", which is not a fair comparison.
- In `run_all_non_sep.py` the condition is a conjunction over min/avg/furthest/
  slacks, so a single failing variant discards the seed for all of them.
- The actual seeds used are never written out, so the tables are not
  reproducible even in principle.
- The failure rate — the quantity that best characterises the method's stated
  limitation — is silently discarded.

**Demonstrated**, Breast Cancer / SVM-rbf, seeds 0–14:

- with `USE_TWO=False`, no seed fails → the run uses seeds 0–9;
- with `USE_TWO=True`, **seed 4 fails** → the run uses seeds {0,1,2,3,5,…,10}.

That single substitution is the entire reason the *Baseline* accuracy moves
from .917 to .912 between the two configs in the table above — a flag that
cannot possibly affect the baseline classifier changes the baseline's reported
score, purely through which seeds survive the filter.

Fix: run a **fixed** seed list (e.g. `range(30)`), record per-seed raw results,
and report `is_fit` failures as an explicit column ("solved 24/30") rather than
by exclusion.

### 6.2 Only aggregated LaTeX survives

`write_results` collapses runs to `mean ± std` and writes a `.tex` fragment.
The per-seed numbers are thrown away, so no significance testing
(Wilcoxon/Friedman–Nemenyi) is possible after the fact. Write raw per-seed CSV
first, derive tables second.

### 6.3 Global mutable config

`deltas/misc/use_two.py` holds `USE_TWO`, `USE_GLOBAL_R`, `RANDOM_STATE` as
module-level constants read at import time all over the package. The
`results-two` / `results-two2` ablations were produced by hand-editing this
file. Nothing in a results file records which setting produced it. Promote
these to explicit arguments (or at minimum stamp them into the results header).

### 6.4 Numbers are truncated, not rounded

`write_results` formats with `str(mean)[1:5]`, so `0.6729 → .672`. Every table
in both papers is systematically biased low in the last digit. Use `f'{x:.3f}'`.

### 6.5 Table stitching is positional

`combine_tables` indexes `lines[3][:20] + 'l' + …` and hardcodes line ranges
(`range(7, 13)` for ECAI, `range(7, 17)` for non-sep). It breaks if pandas
changes `to_latex` output or if the number of methods changes. Currently works
on pandas 2.2.1.

### 6.6 Non-deterministic / networked loaders

- `breast_cancer_W.get_Wisconsin_breast_cancer` calls `shuffle_data(data)`
  **without the seed** → different split every run. (Not used in the paper's
  tables — the paper's "Breast Cancer" is `sklearn_toy.get_breast_cancer` — but
  it is in the notebooks' dataset menu.)
- `heart_disease.get_HD` calls `fetch_ucirepo(id=45)` over the network on every
  single call, so a 10-seed run makes 10 HTTP requests and will fail offline.
  Cache it to `deltas/data/datasets/heart_disease/`.

### 6.7 `run_all_non_sep.main()` is disabled

The experiment loop is commented out; only `combine_tables` runs. Uncomment
before trying to regenerate the non-separable tables.

### 6.8 Environment is not pinned

Packaging moved to pdm in `c1c0a96` (`pyproject.toml`; `setup.py` and
`requirements.txt` deleted). `pyproject.toml` lists the same dependency names
with **no version bounds**. The env that works is conda `deltas`
(python 3.10.13, sklearn **1.3.2**, numpy 1.26.4, pandas 2.2.1, scipy 1.11.4,
imblearn 0.12.0, torch 2.1.1). sklearn must stay on 1.3.x — see §7.1.
`costcla` is only needed by `deltas/data/loaders/costcla.py` (Credit Scoring /
Direct Marketing datasets, unused in both papers); `deltas/costcla_local/` is a
self-contained vendored copy that provides BMR and Thresholding. Consider making
`costcla`, `torch`, `umap-learn` optional extras — they are the three most
fragile dependencies and none is needed to reproduce either paper's tables.

---

## 7. Bugs and code issues

### 7.1 Real bugs

| # | Where | Issue |
|---|---|---|
| B1 | `optimisation/optimise_deltas.py:82` | `J[constraints != 0] = max_loss + 1` uses exact float equality to filter the grid. Measured on a solvable projection: only 7508 of 10000 grid points have an exactly-zero residual; the other 2492 sit at ~1e-16 and are **discarded as constraint violations**. `tol_constraint = 1e-6` is defined on line 69 and never used (the correct lines are commented out just above). Change to `np.abs(constraints) > tol_constraint`. |
| B2 | `model/downsample.py:88–95, 161` | `support_max_hit` is only assigned inside `if method in methods_supported`, but is read in the `len(losses) == 0` error branch. It happens to be safe today only because the read site is `if method not in methods_supported or support_max_hit == False` and Python short-circuits. Any reordering of that condition gives an `UnboundLocalError`. The method-name validation that would make this unnecessary is commented out at line 60, so an unrecognised `method` string silently falls through to the `'supports' in method` checks instead of raising. |
| B3 | `data/loaders/breast_cancer_W.py:30` | `shuffle_data(data)` missing `seed=seed` → non-reproducible split. |
| B4 | `model/non_sep.py:77-79` | `np.delete(line, 0)` / `np.delete(line, -1)` results discarded (numpy is not in-place). Harmless — `get_valid_linspace` already removed invalid points — but it means the intent is unimplemented. |
| B5 | `notebooks-*/run_all*.py` `write_results` | `str(means[i])[1:sf]` truncates instead of rounding, and produces garbage for any value `≥ 1.0` or negative. |
| B6 | `pipeline/evaluation.py:130` | `if xp1.shape[0] > xp2.shape[1]:` — compares a count against a *dimension*. Almost certainly meant `xp2.shape[0]`. Only affects plot colour/marker ordering. |
| B7 | `utils/radius.py:24` | `calc_emp_R` is dead code that `print`s and returns `None`. |
| B8 | `model/downsample.py:73` | `max_trials = min(len(y)//2, max_trials)` silently caps the budget for support-based methods, so passing `max_trials=10000` does not do what it says. Undocumented. |

### 7.2 Smells worth cleaning

- `deltas/pipeline/pipeline_old.py` (296 lines) is superseded — delete or move to `dev/`.
- `notebooks-ECAI/run_all_two copy.py` differs from `run_all_two.py` by two
  string constants. Parameterise instead.
- `deltas/classifiers/models.py` vendors ~350 lines of sklearn `MLPClassifier`
  internals to add sample weighting. This is the single most fragile piece of
  the codebase. sklearn's `MLPClassifier` still has no `sample_weight`, so
  the vendoring is justified, but it should be isolated, version-guarded and
  covered by a test.
- No test suite at all. `.pytest_cache` records exactly one collected node —
  `notebooks-ECAI/run_all.py`, which failed. The equations module in particular
  (`delta2_given_delta1_matt` vs `contraint_eq7` vs `J_derivative`) is
  self-checkable: a handful of property tests (constraint residual ≈ 0 for the
  derived `δ₂`; analytic gradient vs finite differences) would have caught B1.
- `deltas/__init__.py` is `from .data import utils`, and loaders then call
  `deltas.data.utils.shuffle_data(...)` relying on that import side-effect, with
  `# type: ignore` sprinkled to silence the resulting warnings. Import directly.

---

## 8. Caching (the stated second direction)

Measured on this machine (2026-08-07, conda `deltas` env):

| stage | Breast Cancer | MIMIC-III mortality |
|---|---|---|
| load data | 0.01 s | 0.03 s |
| train the 3 baselines (baseline / BW / SMOTE) | 1.6 s | **270 s** |
| slacks deltas fit (`max_trials=10000`) | ~1 s | 21 s |
| non-sep deltas fit (`min`) | 0.9 s | 14 s |

So MIMIC is ~5 min/seed of which **88% is classifier training**, ×10 seeds
≈ 50 min per experiment script — and there are five scripts
(`run_all`, `run_all continuous`, `run_all_two`, `run_all_two copy`,
`run_all_non_sep`) that all retrain the identical models from scratch.
That is roughly 4 hours of pure redundant training to regenerate the tables.

**Recommendation: cache the projection, not just the model.** Everything
downstream of training only needs 1-D arrays:

- deltas needs `clf.get_projection(X_train)` and `y_train`;
- evaluation needs each baseline's `predict(X_test)` and `predict_proba(X_test)`.

So a cache entry per `(dataset, seed, scale, model, config-hash)` holding

```
X_proj_train, y_train, X_proj_test, y_test,
{name: (preds_test, proba_test) for each baseline classifier}
```

makes the entire results table recomputable with no model in memory and no
sklearn/torch involvement. Method development on the deltas side then costs
milliseconds. Store as `.npz` under a gitignored `cache/`, keyed by a hash of
the config dict, and stamp `USE_TWO` / `USE_GLOBAL_R` / library versions into
the key so a stale cache cannot silently poison results.

Second tier: `joblib.dump` the fitted estimators themselves under
`cache/models/` for the cases where a plot needs a full decision surface
(`plots.plot_decision_boundary`).

Third tier: cache the *datasets* — mainly `heart_disease` (network fetch) and
the MIMIC CSV parse.

---

## 9. Suggested order of work

0. **Set `USE_TWO = True`** (it is the correct setting — see §5A) and stop it
   being a hand-edited global. Right now the two papers are on opposite
   settings and neither records it, which blocks every comparison. (§5A, §6.0)
1. **Reproducibility harness** — fixed seed list, raw per-seed CSV, config
   (including the flags) recorded in the output, `is_fit` failures reported not
   hidden. (§6.1, §6.2)
2. **Projection cache** (§8) — unblocks everything else by making iteration fast.
3. **Fix B1 and B2**, add property tests on `utils/equations.py`. Re-run the
   ECAI tables and check they still match `results-locked/`. If they move,
   that is itself a finding.
4. **Loss-form ablation** on the non-separable method (§3). One-line change.
5. **Overlap sweep figure** (§4, experiment 1) using the ECAI method — establish
   the failure mode you are trying to fix before fixing it.
6. **Implement approach A** (Clopper–Pearson) as a new `deltas/model/` estimator
   alongside `non_sep`, since it needs no `R` and is never infeasible. Show it
   recovers the ECAI boundary in the separable limit.
7. **Bound-vs-actual plots** (§4, experiment 3) — the biggest missing piece for
   a theory paper.
8. Statistical tests (Friedman + Nemenyi over datasets), 30+ seeds.

Items 1–2 are engineering, 3–5 are a week, 6–8 are the paper.

**Status as of 2026-08-07** (see §10): 1 ✅, 2 ✅ (classifier cache, which is
what actually mattered), 5 ✅, 6 ✅ (both Clopper–Pearson and DKW), 7 ✅
(§10.6 — and it found a real validity problem), 8 partially ✅
(`experiments/significance.py`: Wilcoxon + average ranks + Friedman; still on
10 seeds, not 30). Still open: **0** (`USE_TWO` is still a hand-edited global —
only *recorded* now, not parameterised), **3** (B1/B2 unfixed, no test suite),
**4** (the non-sep loss-form ablation is still a one-line change nobody has
run).

---

## 10. Update — what was built and what it showed (2026-08-07)

### 10.1 Caching (`deltas/utils/cache.py`, `deltas/pipeline/cached.py`)

Disk cache keyed by a SHA-256 of `(kind, config, {sklearn, numpy, cache_version})`,
so a stale entry cannot silently poison results. Two namespaces, `dataset` and
`classifiers`. Verified **bit-identical predictions** against the uncached path
on all five baseline classifiers.

The classifier key deliberately excludes `USE_TWO` / `USE_GLOBAL_R`: baselines
are trained from data alone, so one training run serves every deltas config.

Measured: MIMIC-III drops from **270 s to 0.0 s** per seed on re-run; Breast
Cancer 3.3 s → 0.0 s. Datasets whose loaders hit the network (`heart_disease`
→ `fetch_ucirepo`) are now fetched once.

`cache.sanitise()` replaces unpicklable values with their string form — needed
because `heart_disease.get_HD` stores a live `ucimlrepo` metadata object in
`feature_names`/`description`, which `joblib` cannot serialise.

### 10.2 Reproducible runner (`experiments/run_experiments.py`)

Fixed seed list, per-seed raw CSV, `is_fit` failures recorded as NaN and
reported as a `solved n/10` column, config stamped into the LaTeX comments and
a `config.json`, values **rounded** not truncated.

The fix is immediately visible: with seeds pinned to 0–9 the baselines now
reproduce the published numbers exactly (Breast Cancer Baseline .917/.917/.914
vs published .917/.917/.913; Hepatitis .828/.624/.486 vs .827/.623/.485 — the
last digit is the old truncation bug). Under the old seed-search they moved
with `USE_TWO`, a flag that cannot touch a baseline.

### 10.3 ⚠️ The slack method fails far more often than the paper suggests

With `USE_TWO=True` (the correct setting, §5A) on fixed seeds 0–9:

| Dataset | Slacks Deltas solved |
|---|---|
| Pima Diabetes | **0/10** |
| Heart Disease | **4/10** |
| Hepatitis | 8/10 |
| Gaussian | 8/10 |
| Breast Cancer | 9/10 |

The old runner searched seeds until 10 *successes*, so this never appeared in
any table — and on Pima it would have had to search a long way, or hang. Every
published "Slacks Deltas" row is conditioned on the method having worked.
This is the strongest available evidence for the overlap-native reformulation.

### 10.4 Overlap-native estimators (`deltas/model/overlap.py`)

`binomial_deltas` (Clopper–Pearson, exact, with a union bound over the `N+1`
distinct thresholds) and `dkw_deltas` (one-sided DKW, uniform in `b` for free).
Both validated by simulation: CP coverage 0.965–1.000 against nominal 0.95,
DKW 0.999–1.000; at `m=0` CP gives `1-δ^(1/N)`, matching `ln(1/δ)/N` to three
figures, confirming the separable case is the `m=0` corner.

Neither uses `R` or the concentration inequality on the mean. **Never
infeasible** — `solved 10/10` on every dataset. Fitting takes 0.2 s
(Clopper–Pearson) / 0.004 s (DKW) vs 21 s for the slack method on MIMIC.

**Two decision rules, and the sum rule is a trap.** `L_i(b)` depends on `b`
only through the count `m_i(b)`, so when `N_2` is small it is flat over long
stretches; minimising `L_1 + L_2` then spends that freedom on the majority and
pushes the boundary *towards* the minority. On the synthetic Gaussian
(`N_1=1000, N_2=10`): sum rule `b=+1.07`, test balanced error **0.192**;
minimax `b=-0.53`, **0.102**; test optimum `b=-0.13`, 0.072.

The honest reading: at `N_2=10` a distribution-free certificate cannot resolve
the minority error at all — every boundary keeping `m_2=0` carries the same
certificate. This is a real limit of the fully distribution-free route, and it
is precisely what the original `R/√N` geometry was buying: an extrapolation
assumption about where the minority mass could be beyond the observed points.
Minimax recovers most of the lost behaviour without reintroducing it.

### 10.5 Where the new methods stand

Final numbers, after the §10.7 tie-break fix and a full re-run at
`USE_TWO = True`, ten fixed seeds:

| method | avg rank (G-Mean) | mean G-Mean | ever fails? |
|---|---|---|---|
| Min Deltas | 2.33 | .761 | yes (3/10 Pima, 1/10 Hep) |
| F Deltas | 2.67 | .763 | yes (same) |
| CP minimax | 3.83 | .760 | **no** |
| DKW minimax | 3.83 | .762 | **no** |
| Slacks Deltas | 5.83 | .774 † | yes (10/10 Pima, 6/10 HD) |
| DKW sum | 6.92 | .737 | no |
| Thresholding | 8.25 | .737 | no |
| CP sum | 8.75 | .732 | no |
| Baseline | 12.17 | .516 | no |

† averaged only over datasets it solved — Pima is excluded from its mean, which
flatters it. Friedman over the 13 methods that solve everywhere:
χ² = 49.3, p < 1e-4.

`CP minimax` is the best method on **Pima Diabetes** across all three metrics
(.702/.699/.691 vs Thresholding .679/.657/.618) — and no other deltas variant
solves it at all. It is within .001 of the best G-Mean on Heart Disease (.847
vs .848). On Breast Cancer and Hepatitis it is competitive but behind the
slack/`Min Deltas` variants.

Honest summary: **the overlap-native methods do not win on average — they rank
third and fourth — but they are the only certified methods here that never
fail.** That is the claim the draft makes, and it is the defensible one. The
two measures are near-interchangeable in practice: CP and DKW minimax tie on
average rank and pick boundaries within 0.03 of each other on the synthetic
diagnostic, because minimax depends only on where the two per-class curves
cross and both bounds are monotone in the same count. DKW is ~50× faster to
fit (0.004 s vs 0.2 s); CP is tighter at small `m`. Prefer CP for reporting a
certificate, DKW if fit time matters.

Raw per-seed results in `experiments/results/raw/*.csv`, tables in
`experiments/results/`, figures in the Overleaf draft. Regenerate everything
with `experiments/reproduce.sh`.

### 10.6 ⚠️ The certificate is optimistic on training data — and so is the published method's

Because the overlap methods report the bound they certify, it can be checked
against measured test error. Over 4 datasets × 10 seeds × 2 classes
(`experiments/validate_bounds.py`, raw in `results/bound_validation.csv`):

| Certificate computed on | observed coverage | nominal |
|---|---|---|
| the training data | **0.81** | 0.96 |
| a 35% held-out calibration split | **1.00** | 0.90 |

Per dataset the training-data coverage is Breast Cancer 0.90, Heart Disease
0.95, Pima **0.75**, Hepatitis **0.65**.

**Cause.** The binomial model `m_i(b) ~ Binomial(N_i, e_i(b))` needs the
projected training points to be i.i.d. draws from the projected
class-conditional law. They are not: the classifier that *defines* the
projection was fitted to those same points and has pushed them to the correct
side, so `m_i` under-counts. Verified — the violation tracks classifier
optimism exactly:

| Dataset | train err | test err | optimism | coverage |
|---|---|---|---|---|
| Breast Cancer | .054 | .067 | .013 | 0.90 |
| Heart Disease | .121 | .148 | .027 | 0.95 |
| Pima Diabetes | .177 | .298 | .121 | 0.75 |
| Hepatitis | .093 | .301 | **.208** | 0.65 |

Ruled out as the cause: the data-dependent choice of `δ`. Repeating with a
*pre-specified* `δ ∈ {0.05, 0.10}` and the threshold union bound still gives
0.85/0.83 coverage.

**Fix.** Sample splitting — fit the classifier on one part, compute the
certificate on a held-out part, as conformal prediction does. Restores coverage
to 1.00 on every dataset, at the cost of a looser bound (mean `U_i` .32 → .52).

**This applies to the published method too.** `R̄ᵢ` is likewise the empirical
support of the classifier's own training data, so the ECAI paper's `1−δᵢ`
confidence statements carry the same optimism. It has never shown up because
that method never reports the quantity it certifies — there is nothing to check
it against. Worth a paragraph in any journal version, and it is an argument for
adding a calibration split to the pipeline generally.

### 10.7 A tie-break bug the tests caught (and why it mattered)

The minimax rule's arg-min is a *plateau*, not a point (that is the whole point
of §10.4). The first implementation broke the tie by taking the middle
**index** of the plateau. A mirror-symmetry property test
(`tests/test_overlap.py::test_orientation_is_symmetric`) failed: reflecting the
data about zero did not reflect the boundary, because the middle index depends
on how the projected points happen to be spaced.

Fixed by taking the midpoint of the plateau **in value**. This is both
symmetric and better motivated — every boundary on the plateau carries an
identical certificate, so the remaining freedom should buy the largest distance
to the training points at either end.

It is not a cosmetic fix. On the synthetic Gaussian the minimax test balanced
error drops from **0.102 to 0.0735** (Clopper–Pearson) and **0.0720** (DKW),
against a test optimum of 0.0715 — i.e. from "recovers most of the achievable
performance" to "essentially optimal". All results in §10.5 and the draft were
regenerated after this fix.

Worth noting the general lesson: this is the kind of defect that no amount of
staring at a results table surfaces, and the repo had no test suite at all
(§7.2). `tests/test_overlap.py` now covers the bound coverage by simulation,
the `m = 0` separable corner, always-solvability across seven edge cases, and
the two symmetry/behaviour properties the write-up claims.

### 10.8 Power-cut recovery note (2026-08-07)

A power cut killed the MIMIC run mid-way (seed 3 of 10). Recovery was clean and
worth recording, because it validates two design choices:

- **No cache corruption.** `cache.save()` writes to `<key>.joblib.tmp` and then
  `os.replace()`s it, which is atomic on POSIX, so an interrupted write can
  never leave a half-written entry that looks valid. After the cut: 110/110
  entries loaded, zero stray `.tmp` files.
- **No lost work.** The MIMIC classifiers for seeds 0–3 were already cached, so
  the restart only had to train seeds 4–9. Under the old scripts the whole
  ~50 min would have been repaid.

If a future cut *does* leave a bad entry, `cache.load()` catches any exception
and returns `None` (treated as a miss), so a corrupt file degrades to a
recompute rather than a crash. To force a clean slate:
`python -c "import deltas.utils.cache as c; c.clear()"`, or
`FRESH=1 ./experiments/reproduce.sh`.

---

## 11. The wide grid: 32 datasets × 7 models (2026-08-09)

Built to answer two questions the six-dataset study could not: **is the method
really classifier-agnostic**, and **how bad is the certificate optimism of
§10.6 in general?**

Setup: `experiments/export_projections.py` (sibling venv) →
`experiments/run_wide.py` → `experiments/analyse_wide.py`. 32 datasets
(4 synthetic, 26 tabular, 2 MedMNIST), 7 model families, 10 seeds, and every
cell run twice — once with the certificate computed on the training data
(`naive`) and once on a 35% held-out calibration split (`split`).
**48,862 rows, 35,536 certificate observations.** Raw: `results/wide.csv`.

### 11.1 ⭐ The certificate optimism is much worse than the 4-dataset study showed

| certificate computed on | coverage | nominal |
|---|---|---|
| the training data | **0.684** | 0.948 |
| a 35% calibration split | 0.949 | 0.878 |

§10.6 reported 0.81 on 4 datasets × 1 model. Across the full grid it is
**0.684** — the certificate is wrong roughly one time in three. Split
calibration restores validity everywhere (0.949 observed ≥ 0.878 nominal).

Robust to test-set size (coverage is *measured* against the test set, so cells
with tiny test sets resolve it poorly):

| restriction | datasets | naive | split |
|---|---|---|---|
| all cells | 32 | 0.684 | 0.949 |
| ≥ 50 test points per class | 25 | 0.702 | 0.957 |
| ≥ 100 test points per class | 13 | 0.687 | 0.955 |

### 11.2 ⭐ The shortfall tracks classifier optimism monotonically

The mechanism claim of §10.6, now on 17,768 observations per mode:

| classifier optimism | naive coverage | split coverage | n (naive) |
|---|---|---|---|
| ≤ .02 | 0.821 | 0.956 | 6664 |
| .02–.05 | 0.867 | 0.960 | 1272 |
| .05–.10 | 0.821 | 0.955 | 2096 |
| .10–.20 | 0.699 | 0.975 | 2280 |
| > .20 | **0.413** | 0.929 | 5456 |

Nominal is ~0.95 in every naive bucket, so the last row is a factor-of-two
shortfall. The split fixes every bucket, including the worst.

### 11.3 ⭐ Coverage by model — and why `NearestClassMean` is the control

| model | naive | split |
|---|---|---|
| NearestClassMean | **0.891** | 0.941 |
| Linear | 0.798 | 0.953 |
| MLP | 0.792 | 0.949 |
| LDA | 0.693 | 0.944 |
| SVM-rbf | 0.571 | 0.950 |
| RandomForest | 0.553 | 0.955 |
| GradientBoosting | **0.486** | 0.948 |

This is the cleanest confirmation of the diagnosis in the whole study. The
ordering is essentially the ordering of how hard each family overfits its
training data. `NearestClassMean` — which barely overfits, and whose
centroid-plus-radius geometry is *exactly what the published ECAI derivation
assumes* — is the least broken. `GradientBoosting`, which drives training
error towards zero, is the most broken. **After splitting every model lands in
0.94–0.96, i.e. the fix is model-independent.**

### 11.4 The overlap methods now rank first outright

Average rank by G-Mean over 224 (dataset, model) cells, `naive`:

| method | avg rank | mean G-Mean | cells solved |
|---|---|---|---|
| DKW Minimax | **3.25** | .716 | 224/224 |
| CP Minimax | 3.48 | .716 | 224/224 |
| Min Deltas | 4.75 | .707 | 204/224 |
| SMOTE | 5.08 | .676 | 224/224 |
| F Deltas | 5.28 | .697 | 204/224 |
| Slacks Deltas | 6.10 | .712 | 171/224 |
| Threshold | 6.42 | .659 | 224/224 |
| DKW Sum | 6.72 | .647 | 224/224 |
| CP Sum | 7.12 | .625 | 224/224 |
| Baseline | 8.76 | .471 | 224/224 |
| Balanced Weights | 9.04 | .608 | 96/224 |

On the six-dataset study (§10.5) `Min`/`F Deltas` edged ahead of the overlap
methods. On the wide grid they do not — **CP/DKW Minimax are first and second**,
and are the only certified methods that solve every cell.

Solve rates:

| method | naive | split |
|---|---|---|
| CP / DKW (both rules) | **1.000** | **1.000** |
| Min / F Deltas | 0.808 | 0.343 |
| Slacks Deltas | 0.584 | 0.244 |

The published slack method fails on **42% of the grid**, and on 76% of it once
a calibration split shrinks the training set. The overlap methods have no
feasibility condition, so this cannot happen to them.

### 11.5 ⚠️ The "cost of calibration" table has a survivorship trap

| method | naive | split | Δ |
|---|---|---|---|
| CP Minimax | .718 | .694 | −.024 |
| DKW Minimax | .718 | .696 | −.022 |
| Threshold | .662 | .635 | −.027 |
| CP Sum | .628 | .522 | −.106 |
| Slacks Deltas | .717 | .862 | **+.144** |
| Min Deltas | .720 | .865 | **+.145** |
| F Deltas | .709 | .864 | **+.156** |

**The three apparent *improvements* are artefacts, not results.** Those methods
solve only 24–34% of split cells, and the cells they still solve are the easy
ones (well separated, plenty of minority data). Their split means are computed
over a favourable subset. Any comparison of `naive` vs `split` must be
restricted to methods with equal solve rates, which in practice means the
overlap methods, the baselines, and `Threshold`.

Read on the methods that always solve, calibration costs about **0.025 G-Mean**
— cheap for a certificate that is actually true. `CP Sum` loses four times as
much, consistent with the sum rule's known degeneracy (§10.4) getting worse as
`N_cal` shrinks.

### 11.6 A design mistake worth recording

The first pass sized the train/test split as "thin the training minority to
10:1, leave ≥10 minority points for test". On the datasets with the *largest*
minority pool that took almost all of it: **Stroke Prediction ended up with 10
test points per class**, so the measured error could only be a multiple of 0.1
and its coverage number was mostly quantisation noise. Those cells were the
only ones showing split-mode "violations" (Stroke −0.248, Cervical Cancer
−0.070, Thyroid Sick −0.004 against nominal).

Fixed by capping the training minority at half the available minority
(`export_projections.py::load_split`); Stroke goes from 10 to 125 test points
per class while keeping its 19.6:1 training imbalance. Those three datasets
were re-exported and re-run (`merge_wide.py` folds partial runs in).

Lesson: **test-set size is not something to trade for training imbalance when
the quantity being measured is a coverage probability.**

### 11.7 What this means for the papers

1. The **classifier-agnostic** claim now has evidence: 7 model families
   spanning linear, kernel, neural, tree-ensemble and prototype, on one grid.
2. The **validity** claim needs the calibration split. Without it the reported
   confidence is wrong a third of the time, and the ECAI method inherits the
   same flaw invisibly (§10.6).
3. The **always-solvable** claim is the strongest empirical result here: 100%
   vs 58% for the published method over 224 cells.
4. `NearestClassMean` deserves a paragraph of its own — it is the model the
   original derivation is written for, and it is the one where the untouched
   certificate is least wrong.
