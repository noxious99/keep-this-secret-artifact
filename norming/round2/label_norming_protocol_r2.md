# Label norming protocol, round 2 (Phase 1c, model-based)

*Written 2026-11-10, before any round-2 norming data exist. Frozen by the commit that adds it. Round 1 (`norming/label_norming_protocol.md`) failed at the matching step (its "Round 1 outcome" section); this round replaces its candidate list and values. Everything not stated here is as in round 1, including amendments A1–A2 and deviations D2–D5, which are now part of the method. Any change after round-2 data exist is recorded as an amendment or deviation at the end of this file.*

## 1. Why a second round
Round 1's classes separated on meaning (11 sensitive, 10 salient, 11 neutral labels eligible) but could not be matched on form. Sensitive labels were less natural with a 2-digit value and less number-like than the others. No subset met any rung, by the protocol's search or by a broad random search.

A pre-screen of new candidates on the five studied models (`label_prescreen.ipynb`, `results/prescreen_summary.json`) computed **only** the form covariates; sensitivity and importance were not computed. It showed:
- with **4-digit values**, the naturalness gap between classes mostly disappears: sensitive labels fit a 4-digit value better, salient labels a 2-digit one;
- a 12-per-class set balanced on form reaches a largest |SMD| of 0.50 with 4-digit values, against 0.60 with 2-digit values.

## 2. Candidates and values
- **36 labels, 12 per intended class,** in `norming/round2/candidate_labels.json`. They are the 12-per-class set best balanced on form with 4-digit values (`results/prescreen_balanced_sets.txt`), approved by the authors.
- **Template:** `The {label} is {value}.`
- **Values:** 4721, 8306, 1659 (4-digit). Every model-based measure is averaged over the three values.
- **Consequence for v7:** v7 uses 4-digit values; its design is fixed in `preregistration_v7.md`.

## 3. Sources
As in round 1:
- **studied:** Phi-4, Falcon3-10B, Granite-3.3-8B, OLMo-2-32B, Llama-3.1-8B, pinned to their v6 checkpoints;
- **independent:** Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (Kaggle); deepseek-flash and deepseek-v4-pro (API);
- **objective:** token counts, Zipf frequency (`wordfreq` 3.1.1);
- **authors:** the three authors, as a cross-check only.

## 4. Measures
All round-1 measures, computed by `label_norming_r2.ipynb` (`analysis/build_nb_norming.py --round 2`) and `analysis/norming_api_ratings.py --round 2`, with A1, A2, D3 built in. **Changes from round 1:**
1. **Numeric association (primary):** the studied models' own 1–7 rating of "how typical is it for {a/an} {label} to be a number". As with the other local ratings:
   - the digit counts only as a whole answer (A1);
   - a digit after a whitespace token counts (D3);
   - a rating where P(digit) < 0.5 is missing (D4).
   - **Why:** in round 1 this rating agreed with the authors' "typical as a number" judgement (Spearman +0.65), while the probability that a number follows "The ___ is" did not (−0.16; `results/norming_human_vs_model.txt`).
   - The probability measure (A2, D3) is still computed and reported, and used for the sensitivity analysis (§6).
2. **Naturalness** uses the 4-digit values.

API replies are parsed under D2. A reply that repeats a 4-digit value has a first number longer than one digit and is unparsed.

## 5. Scores used for selection
As in round 1, except that the numeric covariate is the z-mean over the studied models of the numeric self-rating (§4.1).

## 6. Selection rule
**Unchanged from round 1 §6:**
- 2-of-3 source eligibility;
- target SMDs ≥ 1.0;
- covariate |SMD| ≤ 0.5;
- the same deterministic removal search;
- the ladder (0.75; then a minimum of 5 per class);
- D5 in the SMD.

**Sensitivity:** the selection is rerun with the numeric covariate as the probability measure. Both results are reported (`results/norming_selection_r2.json`: `sensitivity_num_prob`; `norming/round2/selected_labels.md`). The primary selection is the one used.

**If round 2 fails at every rung,** labels are not chosen by hand:
- v7 uses **all eligible labels** of each class (Step 1), up to 8 per class, keeping the 8 with the most extreme class-defining score (S for sensitive, I for salient, lowest I for neutral; ties alphabetical);
- the covariates that fail the |SMD| ≤ 0.5 check enter v7's analysis as adjustment terms;
- this fallback is restated in `preregistration_v7.md` before any v7 data, and the claim is worded "adjusted for" the covariates, not "matched on" them.

## 7. Author cross-check
- New sheet: `norming/round2/author_ratings_template.csv`. 36 sentences with the value 4721, in a fixed random order (seed 20261111), with no class shown. Each author fills `norming/round2/author_ratings_<initials>.csv` alone; instructions are in `norming/round2/AUTHOR_INSTRUCTIONS.md`.
- **Blindness:** the author working in the design sessions has seen this list grouped by class, and the round-1 and pre-screen results (as in round-1 deviation D1). The other two authors should receive only the CSV and the instructions.
- The author ratings are reported as agreement and never used for selection.

## 8. Order of work and outputs
1. Authors fill the round-2 sheets.
2. `label_norming_r2.ipynb` on Kaggle, once per local model → `results/norming_r2_<model>/norming.json`.
3. `python analysis/norming_api_ratings.py --round 2` → `results/norming_api_r2/`.
4. `python analysis/norming_select.py --round 2` → `results/norming_selection_r2.json`, `norming/round2/selected_labels.md`.

**Code tests before any round-2 data:**
- the round-1 notebook builds byte-identically;
- the round-1 selection reproduces exactly;
- the round-2 selection path passes 6/6 checks on copies of the round-1 outputs;
- the selection self-test passes.

## Amendments and deviations

**Note R2-1 (2026-11-12): author sheets.**
- **Distribution:** the design-session author gave the other two authors the round-2 template (`author_ratings_template.csv`), collected their completed sheets, and added all three (A, B, C) to `norming/round2/`. Each sheet has 36 rows matching the template, with every rating an integer 1–7.
- **Before the sheets were filled:** stage-1 model runs (Granite, Mistral-7B, Phi-4) existed. Only measurement-health statistics were reported from them: digit mass, D4 drops, A1 change, option-order consistency. No label-level or class-level model score was shown, so the §7 rule ("before any model score is shown to them") holds.
- **The design-session author's prior exposure** to the grouped list and to the round-1 and pre-screen results is as stated in §7.
