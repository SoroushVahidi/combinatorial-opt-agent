## NLP4LP Constrained Assignment Results

This document summarizes how the new constrained assignment method performs relative to the existing typed greedy and untyped ablations, using the current `results/paper/nlp4lp_downstream_summary.csv` and `nlp4lp_downstream_types_summary.csv`.

All metrics use **331 test queries** per variant unless otherwise stated.

---

### 1. Main aggregate metrics (orig, TF-IDF)

**Source:** `results/paper/nlp4lp_downstream_summary.csv` (variant=`orig`, baselines `tfidf`, `tfidf_untyped`, `tfidf_constrained`).

| Baseline            | Coverage | TypeMatch | Exact20 (on hits) | InstantiationReady |
|---------------------|----------|-----------|-------------------|--------------------|
| tfidf (typed greedy)        | 0.8222   | 0.2267    | 0.2140            | 0.0725             |
| tfidf_untyped (ablation)    | 0.8222   | 0.1677    | 0.1539            | 0.0453             |
| **tfidf_constrained (new)** | 0.7720   | 0.1980    | **0.3279**        | 0.0272             |

Observations (orig, TF-IDF):

- **Schema R@1** is identical for all three (`0.9063`), since retrieval is unchanged.
- **Coverage** drops slightly for `tfidf_constrained` (~0.82 → 0.77), reflecting more conservative assignments.
- **TypeMatch** for `tfidf_constrained` is **between** typed and untyped (higher than untyped, lower than typed).
- **Exact20_on_hits** improves substantially for `tfidf_constrained` (0.214 → 0.328), indicating better relative-error behavior on the queries where it assigns.
- **InstantiationReady** decreases (0.0725 → 0.0272), because the stricter assignment leaves more slots unfilled and fewer queries reach the 0.8/0.8 thresholds.

In short: constrained assignment trades some coverage and InstantiationReady for **better per-hit numeric accuracy** (Exact20).

---

### 2. Main aggregate metrics (orig, Oracle)

**Source:** `results/paper/nlp4lp_downstream_summary.csv` (variant=`orig`, baselines `oracle`, `oracle_untyped`, `oracle_constrained`).

| Baseline               | Coverage | TypeMatch | Exact20 (on hits) | InstantiationReady |
|------------------------|----------|-----------|-------------------|--------------------|
| oracle (typed greedy)          | 0.8695   | 0.2475    | 0.1871            | 0.0816             |
| oracle_untyped (ablation)      | 0.8695   | 0.1887    | 0.1754            | 0.0453             |
| **oracle_constrained (new)**   | 0.8195   | 0.2052    | **0.3153**        | 0.0211             |

Observations (orig, Oracle):

- **Schema R@1** is 1.0 in all three cases (oracle schemas for every query).
- **Coverage** again drops for `oracle_constrained` (~0.87 → 0.82).
- **TypeMatch** is between typed and untyped.
- **Exact20_on_hits** improves markedly for `oracle_constrained` (0.1871 → 0.3153).
- **InstantiationReady** drops (0.0816 → 0.0211) for the same reason as TF-IDF: stricter, higher-precision assignments result in fewer “fully instantiated” queries under the 0.8/0.8 rule.

Constrained assignment thus **improves numeric accuracy under ideal schemas as well**, but again at the cost of fewer fully instantiated problems.

---

### 3. Variant metrics (noisy, short, TF-IDF)

**Source:** `results/paper/nlp4lp_downstream_summary.csv` (variant=`noisy`, `short`, baselines `tfidf`, `tfidf_constrained`).

#### Noisy

| Baseline            | Coverage | TypeMatch | Exact20 (on hits) | InstantiationReady |
|---------------------|----------|-----------|-------------------|--------------------|
| tfidf (typed)       | 0.7100   | 0.0       | —                 | 0.0                |
| **tfidf_constrained** | 0.2948   | 0.0       | —                 | 0.0                |

Because noisy queries use `<num>` placeholders that are not deterministically resolved to numeric values, **type_match is 0** and **InstantiationReady is 0** for both methods. The constrained method significantly reduces coverage (0.71 → 0.29), reflecting that it often chooses to **leave slots unassigned rather than assign low-confidence placeholders**, but this does not improve downstream type-aware metrics in this variant.

#### Short

| Baseline            | Coverage | TypeMatch | Exact20 (on hits) | InstantiationReady |
|---------------------|----------|-----------|-------------------|--------------------|
| tfidf (typed)       | 0.0333   | 0.0272    | 0.0588            | 0.0060             |
| **tfidf_constrained** | 0.0333   | 0.0393    | **0.3088**        | 0.0060             |

For short queries:

- Coverage and InstantiationReady are effectively unchanged (both very low, as expected from very short inputs).
- **TypeMatch** improves modestly (0.0272 → 0.0393).
- **Exact20_on_hits** improves a lot (0.0588 → 0.3088), indicating better precision where any assignment is made.

---

### 4. Per-type metrics (orig, TF-IDF and Oracle)

**Source:** `results/paper/nlp4lp_downstream_types_summary.csv` (variant=`orig`, baselines `tfidf`, `tfidf_constrained`, `oracle`, `oracle_constrained`).

All per-type metrics are **micro-averaged**:

- Coverage (per type) = `n_filled / n_expected`
- TypeMatch (per type) = `type_correct / n_filled`
- `n_expected` and `n_filled` columns in the CSV give support counts.

#### TF-IDF vs TF-IDF constrained (orig, per-type)

Example (currency and percent types):

- **TF-IDF (typed):**
  - currency: `n_expected=382`, `n_filled=304`, coverage ≈ 0.7958, type_match ≈ 0.3586
  - percent: `n_expected=109`, `n_filled=95`, coverage ≈ 0.8716, type_match ≈ 0.4842
- **TF-IDF constrained:**
  - currency: `n_expected=382`, `n_filled=247`, coverage ≈ 0.6466, type_match ≈ 0.6599
  - percent: `n_expected=109`, `n_filled=65`, coverage ≈ 0.5963, type_match ≈ 0.7846

Interpretation:

- Constrained assignment **fills fewer slots** (coverage down) but **gets types right more often** on the ones it does fill (per-type type_match up significantly, especially for percent).

#### Oracle vs Oracle constrained (orig, per-type)

Example:

- **Oracle (typed):**
  - currency: coverage ≈ 0.8429, type_match ≈ 0.3696
  - percent: coverage ≈ 0.9266, type_match ≈ 0.4950
- **Oracle constrained:**
  - currency: coverage ≈ 0.6466, type_match ≈ 0.6599
  - percent: coverage ≈ 0.5963, type_match ≈ 0.7846

Again, constrained assignment **sacrifices coverage** but yields **much higher per-type type_match**, especially for percent-like slots.

---

### 5. Qualitative per-query examples (orig, TF-IDF)

**Files:**  
- Greedy typed: `results/paper/nlp4lp_downstream_per_query_orig_tfidf.csv`  
- Constrained: `results/paper/nlp4lp_downstream_per_query_orig_tfidf_constrained.csv`  
- Eval text: `data/processed/nlp4lp_eval_orig.jsonl`

#### Example 1: Mrs. Watson’s investment (nlp4lp_test_0)

- **Query snippet:** Mrs. Watson has a total budget (at most \$760000), percent constraints on condos, and a minimum dollar amount on detached houses.
- **Per-query metrics:**
  - Typed TF-IDF: coverage = 0.8, type_match = 0.75, exact20 = 0.25
  - Constrained TF-IDF: coverage = 0.8, type_match = 1.0, exact20 = 1.0
- Interpretation: both methods fill 4/5 scalar slots, but constrained assignment chooses numeric mentions that are **fully type-consistent** and within 20% relative error on all comparable slots.

#### Example 2: Cleaning company advertising (nlp4lp_test_2)

- **Query snippet:** Company chooses numbers of radio vs social media ads under a \$250,000 budget, with min/max bounds on radio and lower bounds on social media.
- **Per-query metrics:**
  - Typed TF-IDF: coverage = 0.75, type_match ≈ 0.17, exact20 = 0.0
  - Constrained TF-IDF: coverage = 0.75, type_match = 0.5, exact20 ≈ 0.33
- Interpretation: coverage is unchanged (6/8 slots), but constrained assignment **triples the fraction of correctly typed slots** and significantly improves relative error on the hit set.

#### Example 3: Breakfast sandwiches (nlp4lp_test_1)

- **Query snippet:** Regular and special sandwiches with egg/bacon usage and profit per sandwich.
- **Per-query metrics:**
  - Typed TF-IDF: coverage = 1.0, type_match = 1.0
  - Constrained TF-IDF: coverage = 1.0, type_match = 0.0
- Interpretation: an example where constrained assignment **hurts**: it rejects all high-precision assignments to avoid low-scoring matches, yielding perfect coverage but zero type_match. This illustrates that the constrained method is more conservative and can fail on otherwise easy instances; it should be reported as such and not oversold.

---

### 6. Overall interpretation

- The constrained assignment method behaves as intended:
  - It preserves retrieval performance (Schema R@1 unchanged).
  - It enforces one-to-one mention–slot assignments using a deterministic global optimization.
  - It tends to **improve numeric accuracy metrics** (Exact20_on_hits and per-type type_match), especially for percent/currency-like slots.
- The method trades this off against:
  - **Lower coverage** (fewer slots filled).
  - **Lower InstantiationReady** (fewer queries reach the 0.8 coverage / 0.8 type_match thresholds).
- For manuscript reporting, this should be presented as:
  - A **more precise but more conservative** assignment method that improves quality on filled slots but does **not** solve the end-to-end instantiation problem.

