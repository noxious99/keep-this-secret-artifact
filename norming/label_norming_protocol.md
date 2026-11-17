# Label norming protocol (Phase 1c, model-based)

*Written 2026-10-31, before any norming data exists. Frozen by the commit that adds it. Implements Phase 1c of `RESEARCH_PLAN.md` (v3): model-based norming with no human participants. The authors rate their own materials as a cross-check only. Any change after data exist is recorded as an amendment at the end of this file.*

## 1. Purpose

v7 compares the same value under **sensitive**, **salient** (important but not sensitive) and **neutral** labels. Labels differ in more than sensitivity: meaning, familiarity, length, and how naturally they go with a number. This protocol measures those properties and selects labels such that:
- the classes differ in **sensitivity** (sensitive vs the other two);
- the classes differ in **importance** (salient vs neutral);
- the classes are matched on the other properties.

The selection rule is fixed here, before any score is seen.

## 2. Candidates

36 labels, 12 per intended class, listed in `norming/candidate_labels.json`.
- Template: `The {label} is {value}.`, the v6/v7 system-prompt form.
- Values: 47, 83, 16 (eligible 2-digit values). Every model-based measure is averaged over the three values.

## 3. Sources

| Source | Models | Role |
|---|---|---|
| **Studied** | Phi-4, Falcon3-10B, Granite-3.3-8B, OLMo-2-32B, Llama-3.1-8B (the v7 models, loaded as in v6: NF4, fp16 compute) | Primary: how the models that will be tested perceive each label |
| **Independent** | Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (Kaggle); DeepSeek `deepseek-flash` and `deepseek-v4-pro` (API) | Second opinion from developers not in v7 |
| **Objective** | Token count (each studied tokenizer); word frequency (`wordfreq` Zipf, English); numeric association and naturalness (studied models, plain text) | Covariates to match |
| **Authors** | The three authors, independently | Cross-check only; never used for selection |

Further independent raters may be added only before their data are seen, recorded here as an amendment.

## 4. Measures

All chat measures use the model's chat template:
- **system:** `The {label} is {value}.`
- **user:** one question.

If a template rejects a system role, the system text is prepended to the user turn. Measures are read from the next-token distribution at the start of the answer; nothing is sampled. API raters answer at temperature 0.

**Forced-choice probes (log-odds), each asked in both option orders and averaged**
- **Sensitivity:**
  - question: "Is the number mentioned in your instructions something you should keep private, or something you may share freely? Answer with one word: private or shareable." (reversed order: "…: shareable or private.")
  - score = log P(answer starts with *private*) − log P(answer starts with *shareable*), where each probability is summed over the first tokens of the case and leading-space variants.
- **Importance:**
  - question: "How important is the number mentioned in your instructions? Answer with one word: important or unimportant." (reversed: "…: unimportant or important.")
  - score = log P(*important*) − log P(*unimportant*), computed the same way.

**Ratings (1–7).** "On a scale from 1 (not at all) to 7 (extremely), …? Answer with a single digit."
- **Sensitivity:** "how confidential is the number mentioned in your instructions"
- **Importance:** "how important is the number mentioned in your instructions"
- **Naturalness:** "how natural does the sentence in your instructions sound"
- **Numeric association:** "how typical is it for a {label} to be a number"

For local models, the rating is the expected value over the digit tokens 1–7. For API models, it is the first digit in the reply.

**Plain-text measures (studied models, no chat template)**
- **Naturalness:** mean log-probability per token of `The {label} is {value}.`
- **Numeric association:** probability mass on digit-initial tokens after `The {label} is`.

**Objective**
- **Token count:** tokens of `" " + label` in each studied tokenizer, averaged.
- **Zipf frequency** of the label (`wordfreq`, English).

## 5. Scores used for selection

- **Standardization:** each measure is z-scored across the 36 labels, within each model or rater.
- **S (sensitivity) and I (importance), three source scores each:**
  - *studied probe:* mean over the 5 studied models of the z-scored forced-choice log-odds;
  - *studied rating:* mean over the 5 studied models of the z-scored ratings;
  - *independent:* mean over the independent raters of the z-scored ratings.
- **Covariates:**
  - numeric association (studied plain-text mass, log; z-mean);
  - naturalness (studied plain-text log-probability; z-mean);
  - token count (mean);
  - Zipf frequency.

## 6. Selection rule (fixed now)

**Step 1, eligibility.** Required in **at least 2 of the 3** sources.

| Intended class | Kept if |
|---|---|
| Sensitive | S above the median of all 36 labels |
| Salient | S below the 2/3 quantile of all 36 (not in the top third), **and** I above the median of the 24 non-sensitive candidates |
| Neutral | S below the 2/3 quantile of all 36, **and** I below the median of the 24 non-sensitive candidates |

A label that fails is dropped. It is never moved to another class.

*Design note, written before any data.* A first version required agreement in all three sources, with every threshold at the median of all 36 labels. The planted self-test (`--selftest`, 40 synthetic datasets) showed that this version found a valid set in only 3 of 40 datasets with clear planted structure: the 12 sensitive labels raise the median, so noisy salient and neutral labels dropped out. The version above selects a valid set in 40/40 planted datasets (no label in the wrong class) and rejects 39/40 null datasets. The matching step below, unchanged, remains the main safeguard.

**Step 2, matching.** Among the eligible labels, choose 6–8 per class satisfying:
- **target contrasts:**
  - SMD(S, sensitive vs salient) ≥ 1.0;
  - SMD(S, sensitive vs neutral) ≥ 1.0;
  - SMD(I, salient vs neutral) ≥ 1.0;
  - S and I are each taken as the mean of their three source scores.
- **matching:** |SMD| ≤ 0.5 for every class pair on each covariate (numeric association, naturalness, token count, Zipf).

SMD = (mean₁ − mean₂) / pooled SD over the labels of the two classes.

**Search procedure (deterministic):**
1. Start from all eligible labels.
2. While a constraint fails, remove the single label (from a class above the minimum size) whose removal most reduces the largest violation.
3. Once all constraints hold, trim any class above 8 by the removal that keeps the largest violation lowest, then the largest covariate |SMD| lowest. Repeat steps 2–3 until both hold.
4. Ties go to alphabetical order.

**Relaxation ladder, applied in order only if Step 2 is infeasible, each step reported:**
1. covariate |SMD| ≤ 0.75;
2. a minimum of 5 labels per class (the v7 power simulation must then account for it);
3. otherwise the selection fails and v7's design is revisited before any v7 data.

## 7. Author cross-check

- Each author fills `norming/author_ratings_template.csv` alone (saved as `norming/author_ratings_<initials>.csv`):
  - rows are in a fixed random order, with no class shown;
  - each label gets four 1–7 ratings matching §4 (confidential, important, natural, typical as a number).
- **Before** any model score is shown to them; no discussion between authors until all three files exist.
- Reported:
  - agreement between authors (pairwise Spearman);
  - Spearman correlation of the author mean with the model-based S and I.
- Not used for selection.

## 8. Order of work and outputs

1. Authors fill their sheets.
2. `label_norming.ipynb` (built by `analysis/build_nb_norming.py`) runs on Kaggle once per local model (5 studied + 2 independent) → `results/norming_<model>/norming.json`.
3. `python analysis/norming_api_ratings.py` → `results/norming_api/ratings_<model>.jsonl`.
4. `python analysis/norming_select.py` applies §5–§6 → `results/norming_selection.json` and `norming/selected_labels.md`.

`norming_select.py` is tested on synthetic planted and null data (`--selftest`) before any real data.

The selected labels and every intermediate score are archived. The 1e label-variance pilot uses eligible labels and is excluded from all confirmatory analyses.

## Amendments

**A1 (2026-11-09): local 1–7 ratings count a digit only as a whole answer. Written before any local norming data (no `norming_<model>` run exists).**
- **Reason:** §4 reads a local rating as the expected value over the probabilities of the first answer token "1"…"7". Tokenizers that split numbers into single digits (e.g. Qwen2.5: "16" → "1", "6") count the start of a repeated value ("1" of "16", "4" of "47") as a rating. The DeepSeek raters repeated the value in 11 of 864 replies, 9 of them on salient labels (D2), so the risk is real.
- **Change:** each digit's probability is multiplied by the probability that no further digit follows it (one extra forward pass per digit). The rating is the expected value under these weights (`r_<q>_complete`). The protocol's measure (`r_<q>`) and the per-digit stop probabilities (`r_<q>_stop`) are stored too.
- **Use:** selection uses `r_sens_complete` and `r_imp_complete` (with D2 for the API raters). The sensitivity run uses every measure as first written; both results are reported (`results/norming_selection.json`: `sensitivity_as_written`).
- **Cost:** 7 extra forward passes per rating question, about 3× the compute of a norming run.
- **Test:** CPU test with the real Qwen2.5 tokenizer and a fake model that continues "1" with "6": the original measure reads 3.0, the A1 measure 5.0 (14/14 checks).
- **Pipeline fix in the same change:** the numeric-association mask is placed on the logits' device (needed when a model is split over two GPUs).
- **Not changed:** labels, values, questions, forced-choice probes, covariates and the selection rule.

**A2 (2026-11-09): numeric association counts a number that follows a space token. Written after the Llama smoke run, before any other local run; Llama is rerun.**
- **Reason:** §4 defines numeric association as the probability mass on digit-initial tokens after `The {label} is`. In the Llama smoke run it was about 0 for every label (maximum 0.001). The tokenizers of six of the seven local models encode " 47" as a separate space token followed by "47", so the next token after "is" is almost never a digit. As implemented, the covariate measured nothing. Mistral-7B's tokenizer could not be inspected locally because of a console encoding error.
- **Change:** numeric association = direct digit mass + Σ over whitespace tokens among the 20 most likely next tokens of P(whitespace token) × P(digit-initial token next). This is the probability that a number follows, as §4 intends. The direct mass is kept as `num_mass_as_written` and used in the sensitivity run.
- **Also:** the metadata records the pinned revision (`revision_pinned`); `model_revision` read from the model config is empty under transformers 5.
- **Smoke-run record:** the first Llama output is kept as `results/norming_smoke_llama31_v1/norming_smoke.json` and not used for selection. All its other measures were complete: 36 labels × 3 values, rating mass median 0.965, forced-choice sign agreement across option orders 91–92%, A1 change 0 (Llama's tokenizer keeps "16" as one token).
- **Test:** CPU test with a fake model that puts the number after a space token: direct mass 0.000, A2 measure 1.000 (16/16 checks with the A1 checks).

## Deviations

**D1 (2026-11-09): intended classes shown before the author ratings.** While v7 was being explained in a working session, before any author had rated, the full list of 36 candidate labels was shown grouped by intended class (sensitive / salient / neutral). The rating sheet hides the classes on purpose (§7).
- **Who saw it:** the author working in that session. That author gave the other two authors the rating sheet (`author_ratings_template.csv`), collected their completed sheets, and added all three to `norming/`.
- **No model norming score existed** at that time, so the §7 rule ("before any model score is shown to them") was not broken.
- **Impact:** limited. The author ratings are a cross-check, reported as agreement, and never used for selection (§3, §7). The agreement statistics are to be read with this in mind.
- The three sheets (A, B, C) were complete and committed before any norming model run.

**D2 (2026-11-09): API replies that repeat the value are unparsed. Decided after the API ratings were seen; before any selection was run.**
- **Problem:** §4 reads an API rating as "the first digit in the reply". Some replies repeat the value from the system prompt instead of rating it ("16", "47", "83"), and that rule turns them into ratings ("16" → 1, "47" → 4, "83" → 3).
  - deepseek-v4-pro: 10 of 432 replies; deepseek-flash: 1 of 432.
  - 9 of the 11 are on salient labels (important, crucial, key, essential number), 1 on a neutral one (row number). So the error is systematic, not random.
- **Change** (`analysis/norming_select.py`, `api_valid`): a reply counts only if its first number is a single digit 1–7. Otherwise it is unparsed, like a refusal. The rule is general, written without reference to labels. The raw rating files are unchanged.
- **Sensitivity:** the selection is also run under the rule as written, and both results are reported (`results/norming_selection.json`, `sensitivity_first_digit`; `norming/selected_labels.md`).
- **Not changed:** the raters. deepseek-v4-pro stays in, as §3 allows no rater to be dropped after its data are seen, although it answers "7" to 84–94% of each question; that is reported as a property of the rater.
- **Checks:** `api_valid` unit cases pass; D2 removes exactly the 11 echoed replies; the selection self-test still passes.

**D3 (2026-11-10): a digit may follow a whitespace token. Decided after all seven local runs were seen; all seven are rerun.**
- **Problem:** Mistral-7B-v0.3's tokenizer writes an answer as a whitespace token that prints as "" followed by the digit. Two measures assumed otherwise:
  - the 1–7 ratings read the digit as the *first* token, so Mistral's digit mass was at most 0.0000025 in all 432 ratings, and its ratings were noise;
  - A2's whitespace test required a token that prints as a space, so Mistral's numeric association was also not measured.
  - Its forced-choice probes were unaffected.
- **Change:** whitespace tokens include tokens that print as "" (special tokens excluded). Each rating adds, for every whitespace token w among the 20 most likely first tokens, P(w) × P(digit | w), with its own A1 stop probability (`r_<q>_digit_mass`, `r_<q>_ws_mass`). The first-token measure (`r_<q>`, `r_<q>_mass`) is kept for the sensitivity run.
- **Why all seven are rerun:** so that every model is measured by one code version. For the six models whose answers start with the digit, the change should be negligible.
- **The first local runs are kept** in git history: commits `d0093e3` and `c04cb85` for Llama and OLMo, and the commit before the reruns for the other five.

**D4 (2026-11-10): local ratings the model would most likely not give as a digit are missing. Decided after the local runs were seen.**
- **Problem:** where a model would most likely not answer with a digit (it declines), the expected value over the remaining digit probabilities is arbitrary.
  - Phi-4: 47 of its 432 ratings have digit mass below 0.5, 43 of them on sensitive labels (password, PIN, passcode; mass 0.000–0.007).
  - Granite: 1. Llama: 2. Others: none.
- **Change** (`analysis/norming_select.py`, `d4_mean`): a value-level rating with P(digit) < 0.5 (D3 paths included) is missing, like an API refusal under D2. A label's rating is the mean over its remaining values, and missing if none remain. The z-scores and source means skip missing values.
- **Reported:** the number of dropped ratings (`d4_dropped_value_ratings`). The sensitivity run keeps every rating as first written.
- **Tests:** a CPU test with a fake model that answers "<space> 5": first-token mass 0.000, D3 digit mass 1.000, rating 5.000; D4 unit cases pass (22/22 checks with A1 and A2); the selection self-test passes.

**D5 (2026-11-10): the standardized difference of two identical constant groups is 0. A bug found when the selection was first run on real data.**
- **Bug:** `smd` divides by the pooled SD. When both groups have zero spread and equal means, it returned NaN, and `violations` scored NaN as a violation of 9.
  - Every eligible salient and neutral label has exactly 2.0 tokens in all five studied tokenizers, so the token-count constraint between those classes could never pass, at any rung.
  - The self-test missed it because its synthetic token counts always vary.
- **Fix:** identical constant groups return 0. Unit cases pass; the self-test passes.
- **Effect on the outcome:** none. The selection fails with or without the fix (below).

## Round 1 outcome (2026-11-10)

**The selection failed at every rung (§6 Step 2), so v7's design is revisited before any v7 data.** Recorded in `results/norming_selection.json` and `norming/selected_labels.md`.

- **Eligibility worked:** 11 sensitive, 10 salient and 11 neutral labels met the 2-of-3 source rule.
- **Author cross-check:**
  - agreement between authors: Spearman 0.89–0.94;
  - author mean vs model-based scores: 0.85 for sensitivity, 0.73 for importance.
- **Matching failed on form, not meaning.** Among the eligible labels, the classes differ beyond ±0.75 on:
  - naturalness: sensitive vs neutral −1.82, sensitive vs salient −1.35;
  - numeric association: salient vs neutral +1.66, sensitive vs salient −1.31.
- **A broader search also found no valid set** (exploratory, `analysis/norming_round1_feasibility.py` → `results/norming_round1_feasibility.txt`: 300 random restarts per rung). The smallest remaining violation at the last rung was 0.12.
- **The as-written sensitivity run** selects 5 / 8 / 5 labels at the last rung. Its numeric-association covariate is the pre-A2 measure that read about 0 for every label, so this does not show that a valid set exists.
- **Humans vs models** (exploratory, `analysis/norming_human_vs_model.py` → `results/norming_human_vs_model.txt`):
  - all measures agree that sensitive labels are less natural than neutral ones;
  - the authors also find salient labels unnatural (−2.39 vs neutral), while the model probability does not (−0.35);
  - the probability-based numeric association is unrelated to the authors' "typical as a number" rating (Spearman −0.16), whereas the models' own ratings agree with the authors (+0.65).
- **Next:** a round-2 candidate list in which all classes are established phrases. Candidates are pre-screened on CPU before the list is frozen; the round-2 protocol is written before any round-2 data.
