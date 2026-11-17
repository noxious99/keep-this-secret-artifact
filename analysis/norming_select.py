"""Applies the frozen label-selection rule (norming/label_norming_protocol.md §5-6) and the author cross-check (§7).

Usage:
  python analysis/norming_select.py              real data: results/norming_<model>/norming.json, results/norming_api/ratings_*.jsonl,
                                                 norming/author_ratings_*.csv  -> results/norming_selection.json, norming/selected_labels.md
  python analysis/norming_select.py --selftest   synthetic planted and null data (no files read or written)"""
import csv, glob, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
STUDIED = ["phi-4", "falcon3-10b-instruct", "granite-33-8b-instruct", "olmo-2-0325-32b-instruct", "llama-31-8b-instruct"]
LOCAL_INDEP = ["qwen25-7b-instruct", "mistral-7b-instruct-v03"]
COVS = ["num", "nat", "tok", "zipf"]
ROUND = 2 if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "2" else 1   # --round 2: norming/round2/label_norming_protocol_r2.md
NDIR = ROOT / "norming" / "round2" if ROUND == 2 else ROOT / "norming"
NUM_MODE = "rate" if ROUND == 2 else "prob"   # numeric association: 'prob' = P(number follows "The ___ is") (round 1; A2 + D3);
                                              # 'rate' = studied models' own 1-7 "typical as a number" rating (round-2 primary; A1 + D3 + D4)
LADDER = [(0.5, 6), (0.75, 6), (0.75, 5)]          # (max covariate |SMD|, min labels per class), applied in order

def z(s): s = pd.Series(s, dtype=float); return (s - s.mean()) / s.std(ddof=0)
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return float("nan")
    if a.var() == 0 and b.var() == 0 and a.mean() == b.mean(): return 0.0      # deviation D5: identical constant groups differ by 0, not NaN
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("inf") * np.sign(a.mean() - b.mean())

# ---------------------------------------------------------------- scores (§5)
def scores(local, indep_ratings, zipf):
    """local: {model: DataFrame(label, cls, fc_sens, fc_imp, r_sens, r_imp, nat_lp, num_mass, token_count)} for studied models;
    indep_ratings: {rater: DataFrame(label, r_sens, r_imp)}; zipf: {label: float}. Returns one row per label."""
    labs = local[STUDIED[0]][["label", "cls"]].copy() if STUDIED[0] in local else next(iter(local.values()))[["label", "cls"]].copy()
    def zmean(frames, col): return pd.concat([z(f.set_index("label")[col]) for f in frames], axis=1).mean(axis=1)
    st = [local[m] for m in STUDIED if m in local]; ind = list(indep_ratings.values())
    out = labs.set_index("label")
    out["S_probe"], out["I_probe"] = zmean(st, "fc_sens"), zmean(st, "fc_imp")
    out["S_rate"], out["I_rate"] = zmean(st, "r_sens"), zmean(st, "r_imp")
    out["S_ind"], out["I_ind"] = zmean(ind, "r_sens"), zmean(ind, "r_imp")
    out["S"] = out[["S_probe", "S_rate", "S_ind"]].mean(axis=1); out["I"] = out[["I_probe", "I_rate", "I_ind"]].mean(axis=1)
    out["num_prob"] = pd.concat([z(np.log(f.set_index("label")["num_mass"].clip(lower=1e-12))) for f in st], axis=1).mean(axis=1)
    if all("r_num" in f.columns for f in st): out["num_rate"] = zmean(st, "r_num")
    out["num"] = out["num_rate"] if NUM_MODE == "rate" else out["num_prob"]
    out["nat"] = zmean(st, "nat_lp")
    out["tok"] = pd.concat([f.set_index("label")["token_count"] for f in st], axis=1).mean(axis=1)
    out["zipf"] = pd.Series(zipf)
    return out.reset_index()

# ---------------------------------------------------------------- selection (§6)
def eligible(df):
    """Step 1 (protocol §6): agreement in at least 2 of the 3 sources.
    sensitive: S above the median of all 36 labels; salient / neutral: S below the 2/3 quantile of all 36 (not in the top third),
    and I above (salient) or below (neutral) the median of the 24 non-sensitive candidates."""
    Ss, Is = ["S_probe", "S_rate", "S_ind"], ["I_probe", "I_rate", "I_ind"]
    ns = df.cls != "sensitive"
    votes = lambda conds: np.sum(conds, axis=0) >= 2
    hiS = votes([df[c] > df[c].median() for c in Ss])
    notTopS = votes([df[c] < df[c].quantile(2 / 3) for c in Ss])
    hiI = votes([df[c] > df.loc[ns, c].median() for c in Is]); loI = votes([df[c] < df.loc[ns, c].median() for c in Is])
    ok = ((df.cls == "sensitive") & hiS) | ((df.cls == "salient") & notTopS & hiI) | ((df.cls == "neutral") & notTopS & loI)
    return {c: sorted(df.label[(df.cls == c) & ok]) for c in ["sensitive", "salient", "neutral"]}

def violations(sel, df, cov_max):
    g = {c: df.set_index("label").loc[sel[c]] for c in sel}
    v = [max(0.0, 1.0 - smd(g["sensitive"].S, g["salient"].S)), max(0.0, 1.0 - smd(g["sensitive"].S, g["neutral"].S)),
         max(0.0, 1.0 - smd(g["salient"].I, g["neutral"].I))]
    for a, b in [("sensitive", "salient"), ("sensitive", "neutral"), ("salient", "neutral")]:
        for cv in COVS: v.append(max(0.0, abs(smd(g[a][cv], g[b][cv])) - cov_max))
    v = [x if np.isfinite(x) else 9.0 for x in v]
    return max(v), sum(v)

def cov_max_abs(sel, df):
    g = {c: df.set_index("label").loc[sel[c]] for c in sel}
    return max(abs(smd(g[a][cv], g[b][cv])) for a, b in [("sensitive", "salient"), ("sensitive", "neutral"), ("salient", "neutral")] for cv in COVS)

def search(elig, df, cov_max, min_n, max_n=8):
    """Deterministic removal (protocol §6 Step 2). While a constraint fails, remove the label (from a class above min_n) whose
    removal most reduces the largest violation; once all constraints hold, trim classes above max_n by the removal that keeps the
    largest violation lowest, then the largest covariate |SMD| lowest. Ties: alphabetical."""
    sel = {c: list(v) for c, v in elig.items()}
    if any(len(v) < min_n for v in sel.values()): return None, "too few eligible labels"
    while True:
        v = violations(sel, df, cov_max)[0]; over = [c for c in sel if len(sel[c]) > max_n]
        if v == 0 and not over: return sel, "ok"
        pool = [c for c in sel if len(sel[c]) > min_n] if v > 0 else over
        if not pool: return None, "constraints not met at the minimum class size"
        best = None
        for c in pool:
            for lab in sorted(sel[c]):
                trial = {k: [x for x in vv if x != lab] for k, vv in sel.items()}
                tv = violations(trial, df, cov_max)
                key = (tv[0], tv[1], lab) if v > 0 else (tv[0], cov_max_abs(trial, df), lab)
                if best is None or key < best[0]: best = (key, c, lab)
        sel[best[1]].remove(best[2])

def select(df):
    elig = eligible(df); log = []
    for cov_max, min_n in LADDER:
        sel, why = search(elig, df, cov_max, min_n); log.append(dict(cov_max=cov_max, min_n=min_n, result=why))
        if sel: return dict(eligible=elig, selected=sel, rung=dict(cov_max=cov_max, min_n=min_n), ladder=log, report=report(sel, df))
    return dict(eligible=elig, selected=None, ladder=log, report=None)

def report(sel, df):
    g = {c: df.set_index("label").loc[sel[c]] for c in sel}
    r = {"S: sensitive vs salient": smd(g["sensitive"].S, g["salient"].S), "S: sensitive vs neutral": smd(g["sensitive"].S, g["neutral"].S),
         "I: salient vs neutral": smd(g["salient"].I, g["neutral"].I)}
    for a, b in [("sensitive", "salient"), ("sensitive", "neutral"), ("salient", "neutral")]:
        for cv in COVS: r[f"{cv}: {a} vs {b}"] = smd(g[a][cv], g[b][cv])
    return {k: round(float(v), 3) for k, v in r.items()}

# ---------------------------------------------------------------- author cross-check (§7)
def authors(df):
    import re
    files = sorted(glob.glob(str(NDIR / "author_ratings_*.csv")))
    files = [f for f in files if not f.endswith("template.csv")]
    if not files: return dict(n_authors=0, note="no author files yet")
    A = {}
    for f in files:
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        A[Path(f).stem.replace("author_ratings_", "")] = {re.match(r"The (.+) is \d+\.$", r["sentence"].strip()).group(1):
                                                          (float(r["confidential_1to7"]), float(r["important_1to7"])) for r in rows}
    names = list(A); labs = list(df.label)
    sens = pd.DataFrame({n: [A[n][l][0] for l in labs] for n in names}, index=labs)
    imp = pd.DataFrame({n: [A[n][l][1] for l in labs] for n in names}, index=labs)
    pair = lambda M: {f"{a}~{b}": round(float(M[a].corr(M[b], method="spearman")), 3) for i, a in enumerate(names) for b in names[i + 1:]}
    d = df.set_index("label")
    return dict(n_authors=len(names), agreement_sensitivity=pair(sens), agreement_importance=pair(imp),
                author_mean_vs_model_S=round(float(sens.mean(axis=1).corr(d.S, method="spearman")), 3),
                author_mean_vs_model_I=round(float(imp.mean(axis=1).corr(d.I, method="spearman")), 3))

# ---------------------------------------------------------------- data loading
def api_valid(reply):
    """Deviation D2 (protocol 'Deviations'): an API reply counts only if its first number is a single digit 1-7; a reply that repeats the
    value from the prompt ('16', '47', '83') is unparsed, like a refusal. The protocol as written takes the first digit 1-7 anywhere."""
    import re
    m = re.search(r"\d+", str(reply)); return bool(m) and len(m.group(0)) == 1 and m.group(0) in "1234567"

D4_MIN_DIGIT_MASS = 0.5; D4_DROPPED = {}
def d4_mean(per_value, q):
    """Deviation D4: a local rating counts only if the model would most likely answer with a digit (P(digit) >= 0.5, D3 paths included);
    otherwise it is missing, like an API refusal. Returns the label's mean over its remaining values (NaN if none) and the number dropped."""
    keep = [v[f"r_{q}_complete"] for v in per_value if v[f"r_{q}_digit_mass"] >= D4_MIN_DIGIT_MASS]
    return (float(np.mean(keep)) if keep else float("nan")), len(per_value) - len(keep)

def load_real(api_parse="d2", local_rating="complete"):
    """api_parse: 'd2' (deviation D2) or 'first_digit' (protocol as written). local_rating: 'complete' (amendment A1: a digit counts
    only as a whole answer) or 'as_written' (protocol §4: expected value over first-token digit probabilities)."""
    local = {}
    pref = "norming_r2_" if ROUND == 2 else "norming_"
    for p in glob.glob(str(ROOT / "results" / f"{pref}*" / "norming.json")):
        if ROUND == 1 and Path(p).parent.name.startswith("norming_r2_"): continue          # round-2 folders never enter round 1
        j = json.load(open(p, encoding="utf-8")); tag = Path(p).parent.name.replace(pref, "")
        d = pd.DataFrame(j["labels"])
        if local_rating == "complete":                                         # A1 + D3 measures, D4 filter; num_mass is A2 + D3
            for q in (("sens", "imp", "num") if ROUND == 2 else ("sens", "imp")):
                m = [d4_mean(pv, q) for pv in d["per_value"]]; d[f"r_{q}"] = [x for x, _ in m]; D4_DROPPED[(tag, q)] = sum(n for _, n in m)
        else: assert local_rating == "as_written", local_rating; d["num_mass"] = d["num_mass_as_written"]
        local[tag] = d
    missing = [m for m in STUDIED if m not in local]
    assert not missing, f"missing studied-model norming runs: {missing}"
    indep = {m: local[m][["label", "r_sens", "r_imp"]] for m in LOCAL_INDEP if m in local}
    for p in glob.glob(str(ROOT / "results" / ("norming_api_r2" if ROUND == 2 else "norming_api") / "ratings_*.jsonl")):
        R = pd.DataFrame(map(json.loads, open(p, encoding="utf-8"))); R = R[R.rating.notna()]
        if api_parse == "d2": R = R[R.reply.map(api_valid)]
        else: assert api_parse == "first_digit", api_parse
        w = R.pivot_table(index="label", columns="q", values="rating", aggfunc="mean").reset_index()
        indep[Path(p).stem.replace("ratings_", "")] = w.rename(columns={"sens": "r_sens", "imp": "r_imp"})[["label", "r_sens", "r_imp"]]
    assert len(indep) >= 2, f"fewer than 2 independent raters: {list(indep)}"
    from wordfreq import zipf_frequency
    zipf = {l: zipf_frequency(l, "en") for l in local[STUDIED[0]].label}
    return local, indep, zipf

def fake(planted, seed=0):
    rng = np.random.default_rng(seed); c = json.load(open(ROOT / "norming" / "candidate_labels.json"))
    labs = [(l, k) for k, v in c["classes"].items() for l in v]
    base = {"sensitive": (1.2, 0.3), "salient": (-0.6, 1.0), "neutral": (-0.6, -1.0)} if planted else {k: (0, 0) for k in c["classes"]}
    def frame(noise):
        rows = []
        for l, k in labs:
            s, i = base[k]
            rows.append(dict(label=l, cls=k, fc_sens=s + rng.normal(0, noise), fc_imp=i + rng.normal(0, noise), r_sens=s + rng.normal(0, noise),
                             r_imp=i + rng.normal(0, noise), nat_lp=rng.normal(-3, 0.5), num_mass=np.exp(rng.normal(-1, 0.5)), token_count=int(rng.integers(1, 4))))
        return pd.DataFrame(rows)
    local = {m: frame(0.5) for m in STUDIED}; indep = {r: frame(0.7)[["label", "r_sens", "r_imp"]] for r in ["rA", "rB", "rC"]}
    return local, indep, {l: float(rng.normal(3.5, 0.8)) for l, _ in labs}

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        ok = True
        for name, planted in [("planted", True), ("null", False)]:
            res = select(scores(*fake(planted)))
            n = {c: len(v) for c, v in res["eligible"].items()}; got = res["selected"]
            print(f"{name}: eligible {n} | selected {({c: len(v) for c, v in got.items()} if got else None)} | rung {res.get('rung')}")
            if got: print("   report:", {k: v for k, v in res["report"].items() if k[0] in "SI"})
            ok &= bool(got) if planted else (got is None)            # null: nothing systematic, so no valid selection
        a, b = [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]
        assert abs(smd(a, b) + 1.0) < 1e-12, "SMD formula"; print("SMD formula ok; selftest", "PASSED" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    local, indep, zipf = load_real("d2", "complete"); df = scores(local, indep, zipf); res = select(df); res["authors"] = authors(df)
    res["independent_raters"] = list(indep); res["scores"] = df.round(4).to_dict("records")
    res["measures"] = ("primary: local ratings as whole answers (A1), digits after a whitespace token counted (D3), ratings with P(digit) < 0.5 "
                       "missing (D4); numeric association after a space token (A2, D3); API replies under D2")
    res["d4_dropped_value_ratings"] = {f"{m}|{q}": n for (m, q), n in sorted(D4_DROPPED.items())}
    if ROUND == 2:                                                          # round 2: sensitivity = numeric association as probability
        res["measures"] = ("round 2 (label_norming_protocol_r2.md): local ratings as whole answers (A1), digits after a whitespace token (D3), "
                           "ratings with P(digit) < 0.5 missing (D4); numeric association = studied models' own 1-7 'typical as a number' rating; "
                           "naturalness with 4-digit values; API replies under D2")
        NUM_MODE = "prob"; s_alt = select(scores(*load_real("d2", "complete"))); NUM_MODE = "rate"
        res["sensitivity_num_prob"] = dict(selected=s_alt["selected"], rung=s_alt.get("rung"), same_selection=s_alt["selected"] == res["selected"])
        res["sensitivity_as_written"] = res["sensitivity_num_prob"]                     # the key used below
        print("sensitivity (numeric association as probability): same selection =", res["sensitivity_num_prob"]["same_selection"])
    else:
        s_alt = select(scores(*load_real("first_digit", "as_written")))    # sensitivity: every measure as the protocol first wrote it
        res["sensitivity_as_written"] = dict(selected=s_alt["selected"], rung=s_alt.get("rung"), same_selection=s_alt["selected"] == res["selected"])
        print("sensitivity (all measures as first written): same selection =", res["sensitivity_as_written"]["same_selection"])
    (ROOT / "results" / ("norming_selection_r2.json" if ROUND == 2 else "norming_selection.json")).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    lines = ["# Selected labels (generated by analysis/norming_select.py; do not edit)", ""]
    if res["selected"]:
        lines += [f"Rung: covariate |SMD| ≤ {res['rung']['cov_max']}, at least {res['rung']['min_n']} per class.", ""]
        for c, v in res["selected"].items(): lines.append(f"- **{c}** ({len(v)}): " + ", ".join(v))
        lines += ["", "| Contrast | SMD |", "|---|---|"] + [f"| {k} | {v:+.2f} |" for k, v in res["report"].items()]
    else:
        lines.append("**Selection failed at every rung**; v7's design must be revisited before any v7 data (protocol §6).")
    sa = res["sensitivity_as_written"]
    verdict = ("**the same selection**." if sa["same_selection"] else "**a different selection**: " + json.dumps(sa["selected"]) + ".")
    if ROUND == 2:
        lines += ["", "Round 2. Measures: " + res["measures"] + ". With numeric association as the probability that a number follows "
                  "\"The ___ is\" (sensitivity): " + verdict]
    else:
        lines += ["", "Measures: local ratings as whole answers (amendment A1), including digits after a whitespace token (D3), with ratings missing "
                  "where the model would most likely not answer with a digit (D4); numeric association including a number after a space token "
                  "(A2, D3); API replies under D2 (a reply repeating the value is unparsed). With every measure as the protocol first wrote it: " + verdict]
    (NDIR / "selected_labels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print("authors:", res["authors"])
