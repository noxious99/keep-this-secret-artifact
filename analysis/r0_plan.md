# R0 plan: reanalysis of existing data (frozen before the scripts)

*Written 2026-10-30, before `r0_audit.py` and `r0_nearest_ref.py` exist. Implements Phase 0 of `RESEARCH_PLAN.md` (v2). Exploratory reanalysis of v5, v5b and v6 data: nothing here changes a pre-registered verdict. Pre-registered scripts and result files are not modified.*

**Blinding status**
- `r0_audit.py` re-derives quantities that were previewed in scratch scripts. It is **not blind**; it is a reproducible robustness check.
- `r0_nearest_ref.py` is **outcome-blind with respect to the nearest-reference metric**: no nearest-reference result has been computed. Exact-match results on the same answers were already seen.

---

## 0. Data and notation

| Run | Models | Cells used |
|---|---|---|
| v5 (`results/pilot_v5_*`) | Qwen3-32B, Mistral-Small-24B | `{none, ord, emph, terse}` × T ∈ {1.0, 0.7, 0.5}; list task |
| v5b (`results/pilot_v5b_*`) | same | `{none, ord, emph, terse}` × T ∈ {0.3, 0.1}; greedy single and batched answers, 80 candidates × 4 instructions |
| v6 (`results/pilot_v6_*`) | Phi-4, Falcon3-10B, Granite-3.3-8B, OLMo-2-32B, Llama-3.1-8B | `list × {none, ref, ord, emph, terse}`, `rep × {none, ord, emph}`, T = 1; greedy 80 candidates × 5 instructions |

**Notation**
- A pair p has values (a, b). Direction 1: the context holds s = a, with decoy s′ = b. Direction 2 is the reverse. Each direction has 3 replicate answers per cell.
- P_s is the model's answer distribution with s in context, at the sampling temperature T.
- Each answer r is scored under both contexts at the same T: LLR(r) = log P_s(r) − log P_s′(r).
- **Pair aggregation** (the pre-registered `pm`): mean over replicates within a direction, then mean over the two directions. The cell estimate is the mean over pairs.
- **All parsing** uses the pre-registered functions in `analysis/v5_analyze.py` (`nums_clean`, `structural`, `verbal`, `digit_disclosure`, `load`), imported by executing the file's header, exactly as `v6_analyze.py` does.

---

## 1. Self-checks (run first; stop on mismatch)

Recomputed from raw `responses.jsonl` + `scores.jsonl`, compared to 1e-9:
- v6 (`results/v6_analysis.json`): P1, P2, ΔKL vs `ord` for {none, ref, emph, terse}, S2 KL, S2 gap, per model.
- v5 (`results/v5_analysis.json`): C1 (clean) and `all_answers/C1`, both models.

`r0_nearest_ref.py` additionally reproduces `v5b_probe.py` item 2: exact-match identification over all sampled answers, `ord`, T = 0.1, Qwen 20.4%, Mistral 12.2% (dossier §18), to 0.05 pp.

---

## 2. Estimands of the instruction contrasts

Sampling distribution = scoring distribution = P_s at the cell's T. This holds because sampling is pure (`top_p = 1`, `top_k = 0`), scoring uses `log_softmax(logits / T)` at the same T with EOS included, and truncated answers (≤ 0.17%) are scored as the exact truncated sequence. The only deviation is fp16 numerics (batched generation vs single-sequence scoring).

**(a) Clean, pre-registered.**
- Mean LLR over answers in the clean event C_s = {r : not structural, verbal or digit disclosure of s}.
- Per direction it estimates E over P_s(·|C_s) of log P_s(r)/P_s′(r).
- This is **neither** the unconditional KL nor the conditional KL. Writing P_s(r|C) = P_s(r)/P_s(C):

      E_{P_s(·|C)} [ log P_s(r)/P_s′(r) ]
        = KL( P_s(·|C) ‖ P_s′(·|C) ) + log P_s(C) − log P_s′(C).      (1)

**(b) All answers.**
- The mean LLR over all answers is an unbiased Monte Carlo estimate of KL(P_s ‖ P_s′), the unconditional KL.

**(c) Selection-corrected clean-conditional KL.**
- From (1): KL(P_s(·|C_s) ‖ P_s′(·|C_s)) = (a) − [ log P_s(C_s) − log P_s′(C_s) ].
- **Pair-specific (primary):**
  - P̂_s(C_s) = the share of the direction's own answers that are clean.
  - P̂_s′(C_s) = the share of the *reverse* direction's answers (generated with s′ in context) that do **not** disclose s, using the same three predicates applied with value s.
  - Unsmoothed.
  - If either estimate is 0, the direction is **undefined**: it is dropped from the pair mean (the pair then uses its other direction) and counted. If both directions are undefined, the pair is dropped and counted.
  - With 3 answers per direction, log P̂ is biased; this is stated, not corrected.
- **Sensitivity 1:** Laplace (k+1)/(n+2) for both probabilities.
- **Sensitivity 2, pooled:** P̂ pooled over all directions in the cell (assumes P_s(C_s) does not depend on the value). Estimand: (a) − log P̄(C)/P̄′(C).

**Contrasts.** P1 = ord − none and P2 = emph − ord (list task); ΔKL for {none, ref, emph, terse} vs ord; terse − none; ref − none (S2). Each is computed under (a), (b), (c) and its sensitivities, and under stratum 4 (§3).

**Uncertainty**
- Pair bootstrap, 10,000 draws, seed 0, resampling the pair-level contrast. Report 95% (pointwise) and 90% (for equivalence) percentile intervals.
- Sign-flip p (pre-registered `signflip`, 20,000).
- For the 7-model P1 table: Holm-adjusted p, plus simultaneous Bonferroni intervals (percentiles 0.05/14 and 1 − 0.05/14).

---

## 3. Exact-occurrence strata

Each answer is classified relative to its in-context value s:

| Stratum | Definition (first match wins) |
|---|---|
| 1 Disclosure | structural OR verbal OR digit-level disclosure (the pre-registered predicates; = not `use`) |
| 2 Exact | s is among the parsed numbers (`nums_clean`) |
| 3 Partial | s occurs inside a longer digit run in the answer (e.g. `430`, `1430`); OR the digit-reversed s is among the parsed numbers; OR two consecutive single-digit runs in the answer spell s |
| 4 None | none of the above |

- **Refusal / task failure** (fewer than 10 parsed numbers) is a separate flag, reported for every stratum.
- **Literal-free analyses** use stratum 4. Their estimand is conditioned on an outcome; the correction of (c) is applied with C replaced by "stratum 4 relative to s". In the decoy-context answers, the stratum is evaluated relative to s.

---

## 4. Containment vs residual: KL chain rule, per pair

Let Z(r) = (1[s ∈ list], 1[s′ ∈ list]), a deterministic function of the answer. By the chain rule, for each direction:

    KL(P_s ‖ P_s′) = KL(P_s^Z ‖ P_s′^Z) + E_{Z∼P_s^Z} [ KL(P_s(·|Z) ‖ P_s′(·|Z)) ]
                   =   containment term   +   residual term                       (2)

- **Total:** estimated by (b), per direction.
- **Containment term:** needs P_s^Z and P_s′^Z for this pair. With 3 answers per direction, plug-in estimates are unusable (zeros give infinite KL). Primary model:
  - **Value-specific base rate β_x:** the probability that number x appears when x is *not* in context. Estimated from all answers in the same cell whose in-context value ≠ x (about 213 answers per value in v6). Jeffreys smoothing (k + ½)/(n + 1).
  - **In-context emission rate ε_v:** the probability that v appears when v *is* in context, from v's 3 answers. Empirical-Bayes beta-binomial shrinkage: a Beta(α, β) prior is fitted by the method of moments to the per-value rates in the cell; the posterior mean is (k + α)/(n + α + β). The prior strength is floored at α + β ≥ 1.
  - **Independence:** presence of s and of s′ are treated as independent given the context. Then P_s^Z = Bern(ε_s) ⊗ Bern(β_s′) and P_s′^Z = Bern(β_s) ⊗ Bern(ε_s′), so

        containment(s, s′) = KL(Bern(ε_s) ‖ Bern(β_s)) + KL(Bern(β_s′) ‖ Bern(ε_s′)).   (3)

  - **Assumptions, stated:** (i) independence, checked by the pooled mutual information between the two indicators, reported; (ii) β_x does not depend on which other value is in context.
- **Residual** = total − containment, per direction, then aggregated with the same pair weights. Negative values are never truncated.
- **Sensitivity:** prior strength × 0.5 and × 2.
- **Descriptive only:**
  - **Pooled-cell estimate:** plug-in KL between the pooled 4-category distribution of Z under the secret context and its swap (exchangeability), with ½-smoothing. Pooling gives mixtures with equal weights on both sides, so by joint convexity it is a **lower bound** on the pair-average containment term (population quantities; plug-in noise aside).
  - **Label-shuffle null floor:** s and s′ swapped at random within each pair-direction for the containment computation only; mean over 200 shuffles.
  - **First-position version (secondary):** Z₂ = (s present, s′ present, first = s, first = s′), pooled-cell only, lower bound.
- **Uncertainty:** pair bootstrap with **the same draws** for total, containment and residual. Base rates and the empirical-Bayes prior are re-estimated inside each draw from the resampled answers. 2,000 draws (the decomposition re-fits per draw; stated).

---

## 5. Equivalence

- 90% pair-bootstrap CI. Outcome classes:
  - **equivalent** if the CI lies within [−m, +m];
  - **meaningfully different** if the CI lies entirely above +m or entirely below −m;
  - **inconclusive** otherwise.
- A separate flag records whether the CI excludes 0.
- **terse − none, KL (a)**, 7 models: no pre-specified margin. m = 0.5 nats is **post hoc**; sensitivity m ∈ {0.25, 1.0}.
- **S2 containment gap, ref − none (pp)**, 5 v6 models: m = 10 pp is a previously specified threshold (v6 S1a "primed") **reused as an exploratory benchmark**; sensitivity m ∈ {5, 15}.

---

## 6. Priming across models (n = 7)

- **Main exploratory test:** Spearman(gap under none, KL (a) under ord), exact two-sided p over all 5,040 permutations.
- **Descriptive only:** Spearman(gap under none, P1) and Spearman(KL none, P1). These share a term (mathematical coupling). For reference, the same statistic is computed after permuting the ord levels across models (5,040 permutations); this shows the coupling-induced distribution and is not a test of priming.
- Low power at n = 7 is stated with every result.

---

## 7. Cross-model tables

- **P1 per model:** estimate; pointwise 95% CI; simultaneous Bonferroni CI; sign-flip p; Holm p over 7.
- **Cochran's Q:**
  - θ_i = P1 estimate of model i; SE_i = SD(pair contrasts)/√n_i;
  - w_i = 1/SE_i²; θ̄ = Σwθ/Σw; Q = Σ w_i (θ_i − θ̄)²; df = k − 1; I² = max(0, (Q − df)/Q);
  - p from the χ² survival function.
  - Assumes independent models and known SEs.
  - Described as heterogeneity among the tested models.
- **Greedy replay, ord − none, per model:**
  - per-candidate scores with the pre-registered rule (`v6_analyze.greedy`);
  - paired over candidates that are non-disclosing in both conditions;
  - mean difference, bootstrap 95% CI over candidates (10,000), sign-flip p, Holm over 7.
  - v5 models use the v5b greedy files.
- **Three-layer table (v6):**
  - L1 replay under `ref`;
  - L2 none − ref;
  - L3 ord − none;
  - each with a bootstrap CI over candidates and sign-flip p.
  - Threat model stated: closed set of 80, greedy references, one greedy observed answer (batched).
- **S3:** the 10 rows from `v6_analysis.json`; the valid denominator (amendment A2); same-sign count; rep-task p per row.

---

## 8. Rates and provenance

- **Per cell:**
  - n;
  - disclosure shares by type;
  - strata 1–4 shares;
  - refusal (< 10 numbers) share;
  - A2 usable share (pre-registered definition).
- **Provenance per run:**
  - every `metadata.json` field relevant to identity: model_id, model_revision, weights_source, load_from, quantization, compute_dtype, torch, transformers, gpu, seeds, prompt_hash, temperatures;
  - each marked *recorded*, *unknown* (e.g. model_revision = None), or *recoverable in Phase 1a*;
  - SHA-256 of every input JSONL used.

---

## 9. Nearest-reference attacker (`r0_nearest_ref.py`)

**Data**
- v5b sampled answers (Qwen3-32B, Mistral-Small-24B), T ∈ {0.1, 0.3}, instructions none / ord / emph / terse; 72 secrets × 3 replicates per group.
- References: `greedy_single.jsonl`, the same instruction, all 80 candidates.

**Primary similarity**
- With o and r the `nums_clean` sequences of the observed answer and the reference:

      sim(o, r) = #{ i < min(|o|,|r|) : o[i] = r[i] } / max(|o|, |r|),   sim = 0 if both are empty.

- **Ties:** score = 1/|argmax set| if the true secret is in it, else 0.
- An empty observed answer ties all 80 candidates, so it scores 1/80.

**Exclusions**
- Primary: stratum-1 answers (disclosure of the true secret) excluded.
- Sensitivity: all answers.
- Reported: the stratum-4 subset.
- The excluded set is fixed by the true labels and kept fixed under permutation.

**Unit and statistic**
- Per-secret score = mean over that secret's retained replicates.
- Group statistic = mean over secrets with ≥ 1 retained answer.

**Chance baseline**
- Within each group (model × instruction × T), permute the true-secret labels across secrets; all replicates of a secret move together.
- Implementation: matrix S[i, j] = mean score of secret i's answers if the truth were j. The null statistic is mean_i S[i, π(i)].
- 20,000 permutations, seed 0; one-sided p = (#{null ≥ observed} + 1)/(N + 1).

**Tests**
- **Primary:** `ord` at T = 0.1 and T = 0.3, both models: 4 tests, Holm.
- **Secondary:** per-secret difference (nearest-reference − exact), with a bootstrap 95% CI over secrets (10,000) and a two-sided sign-flip p.
  - Significant positive → an improvement beyond exact matching.
  - Otherwise → the improvement is inconclusive. If above chance is established, it is stated as such.

**Descriptive**
- Same procedure, without Holm, for:
  - the other instructions;
  - v5 sampled answers at T ∈ {1.0, 0.7, 0.5}, against v5b greedy references (same model and instruction texts; different run, stated);
  - v6 sampled list-task answers (T = 1) against v6 greedy references.
- Other similarity metrics, each used as its own attacker:
  - common-prefix length;
  - Jaccard overlap of the number sets;
  - 1 − normalized Levenshtein distance over the number sequences;
  - length-only (−|len(o) − len(r)|).
- Answer-length distributions (observed and reference) per group.
- Exact-match identification for all groups.

**Threat model, stated in the output**
- Closed set: 80 known candidates.
- The attacker knows the model, the prompt format and the instruction; generates 80 greedy reference answers offline (80 model calls); observes one sampled answer.
- Failure does not show that sampled answers are safe.

---

## 10. Outputs and reproduction

- `results/r0_audit.json` and `results/r0_nearest_ref.json`, both with a `meta` block (seeds, draws, input SHA-256, script name, date) and a `self_check` block.
- Reproduce:
  - `python analysis/r0_audit.py`
  - `python analysis/r0_nearest_ref.py`
- Both are deterministic (fixed seeds).
- R0 is called passed only after review of these outputs and of the derivations in §2 and §4.

---

## Amendment A1 (2026-10-31): after the estimator verification, before any new estimate is computed

**Reason.** `analysis/r0_test_estimators.py` (71/71 unit tests; known-truth simulation, 300 replications × 7 scenarios) showed:
- the estimators of §2 (a), (b), (c) and the §4 decomposition *contrasts* meet the criteria in realistic regimes;
- the stratum-4 corrected estimator (`s4_cond`, §3) is biased by −0.11 to −0.24 nats, with 95% coverage 0.59–0.89, whenever literal-free answers are not the large majority of a cell. Directions without a stratum-4 answer drop out, so the per-direction estimate of P_s(stratum 4) is conditioned on k ≥ 1 and inflated;
- containment *levels* read +0.02 to +0.05 nats high near 0, and the label-shuffle floor of §4 overstates the floor (real-data values 0.03–0.38 nats vs a simulated floor of about 0.05).

**Changes**
1. **Primary "beyond literal occurrence" analysis:** the §4 residual (a chain-rule decomposition that conditions on no outcome). Contrasts in the residual are the primary evidence that an effect is not carried by the secret appearing. `s4` and `s4_cond` are demoted to descriptive.
2. **New sensitivity estimator `s4_cond_eb`:**
   - Same as `s4_cond`, but P_s(stratum 4) is an empirical-Bayes beta-binomial posterior mean, fitted across the values in the cell exactly as ε in §4 (defined for every direction, including k = 0).
   - P_s′(stratum 4 relative to s) is the value-specific base rate: the share of answers in the cell whose in-context value ≠ s that are in stratum 4 relative to s, with Jeffreys smoothing.
   - **Decision rule, fixed now:** `s4_cond_eb` is reported as a validated literal-free estimate only if it meets the original criteria in every realistic simulation scenario (|bias| ≤ 0.05 allowing for Monte Carlo error, coverage 0.92–0.98, null false-positive ≤ 0.07). Otherwise it is descriptive only.
3. **The label-shuffle floor** stays in the output, relabelled "label-shuffle reference (overstates the floor; simulated floor ≈ 0.05 nats)". Containment levels are reported as upward-biased near 0; containment contrasts are the quantity to interpret.
4. **Coverage note:** the pair-bootstrap percentile intervals covered at 0.91–0.96 for a nominal 0.95. Sign-flip p-values, whose false-positive rate was 5–7%, remain the tests of record.

### A1, continued (2026-10-31): wording and reporting; written before the `s4_cond_eb` verdict

**Primary analysis: information beyond exact literal occurrence.**
- The primary analysis uses the KL chain-rule decomposition (§4). It separates the information carried by whether the secret and the decoy appear in the parsed list from the information that remains in the answer once those indicators are accounted for.
- Unlike the literal-free analysis, it retains all answers and averages conditional differences across the presence/absence strata. It therefore does not select answers by a post-generation criterion.
- **The containment component is a model.** It is estimated with the pre-specified product-Bernoulli model (value-specific base rates, empirical-Bayes emission rates). Because the model assumes that secret and decoy presence are conditionally independent and that base rates do not depend on which value is in context, it is treated as an approximation to the exact joint containment distribution.
- **Name of the remainder:** "residual after modelled containment". It equals the exact chain-rule residual only insofar as the approximation is supported. Evidence so far:
  - simulation: a strong-dependence scenario passed, and the base-rate assumption was mildly violated in every scenario with a residual channel, which still passed;
  - real-data mutual information between the two indicators ≤ 0.023 nats.
- **Scope.** The analysis concerns information beyond *exact* secret/decoy list presence. The total includes disclosing answers, so verbal, structural or partial disclosure is not excluded and contributes to the residual.
  - Planned extension, before the paper: the containment variable becomes the full stratum of the secret and of the decoy (disclosure / exact / partial / none), so the residual measures information beyond *any* literal occurrence.
- **Secondary.** The literal-free analysis (stratum 4) is a secondary robustness analysis. `s4_cond` is descriptive, with its counts and undefined cases reported. `s4_cond_eb` is validated or descriptive per the decision rule above.

**Stratum 4 is not Z = (0,0).** Stratum 4 is "no exact, partial or disclosed occurrence of the secret" and does not condition on the decoy. It roughly corresponds to Z ∈ {(0,0), (0,1)}, minus partial and disclosed occurrences.

**Reporting both quantities.**
- The conditional KL within stratum 4 (`s4_cond`, `s4_cond_eb`).
- Its weighted contribution to the decomposition, P_s(stratum 4) × conditional KL (`s4_contrib_eb`; P_s from the empirical-Bayes estimate `s4_p_eb`). With a binary Z (secret present / absent), residual = P_s(absent)·KL_cond(absent) + P_s(present)·KL_cond(present).

### A1 verdict (2026-10-31, after the run): `s4_cond_eb` is descriptive

Simulation (`results/r0_test_estimators.json`, 300 replications per scenario; 74/74 unit tests):
- **Bias:** passes in every realistic scenario (≤ 0.05 nats allowing for Monte Carlo error). In the emission scenario the bias falls from −0.23 (`s4_cond`) to −0.03.
- **Coverage:** fails. 95% intervals cover 0.85–0.91, below the 0.92 floor, and the null false-positive rate is 0.08 (limit 0.07). The likely cause is that the pair bootstrap does not refit the empirical-Bayes prior and base rates inside each draw.

By the decision rule fixed above, `s4_cond_eb` is **descriptive only**. It is not re-tuned after seeing these results. Refitting the correction inside the bootstrap is a candidate for a future, separately validated estimator.

The primary "beyond literal occurrence" evidence remains the residual after modelled containment.
