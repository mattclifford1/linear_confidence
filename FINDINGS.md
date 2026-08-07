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
