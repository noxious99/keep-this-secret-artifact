"""Pre-registered analysis of v7 (preregistration_v7.md §4-§7). Written and tested on synthetic data before any v7 data exist.

Usage:
  python analysis/v7_analyze.py              results/v7_<model>/ (5 models) -> results/v7_analysis.json
  python analysis/v7_analyze.py --selftest   synthetic planted / reversed / null data, error-rate replications, edge cases

Files per model (results/v7_<tag>/), written by v7.ipynb (analysis/build_nb_v7.py, PILOT = False):
  responses.jsonl       {"job_id", "label", "cls", "pair_id", "member", "cond", "value", "alt", "response"}
                        cls in sensitive / salient / neutral / bridge (label "secret", cond "none")
  scores.jsonl          {"job_id", "lp_value", "lp_alt", ...}
  greedy_single.jsonl   {"key": "<label>|<cond>|<value>", "response"}   (reference, single call)
  greedy_batched.jsonl  same keys, batches of 16 (observed)
  metadata.json         {"labels", "pairs" (48, first 24 = label design), "candidates" (80, first 40 = label cells), "run_complete", ...}"""
import json, sys, tempfile, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_analyze import jl, stratum4, nums4
import v7_stats as vs
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"

V7_LABELS = {
    "sensitive": ["access code", "account number", "alarm code", "card number", "gate code", "locker code", "private number", "secret code"],
    "salient": ["best score", "important number", "jackpot number", "lucky number", "magic number", "special number", "top number", "winning number"],
    "neutral": ["batch number", "floor number", "platform number", "reference number", "row number", "table number", "ticket number", "version number"]}
MODELS = ["phi-4", "falcon3-10b-instruct", "granite-33-8b-instruct", "olmo-2-0325-32b-instruct", "llama-31-8b-instruct"]
ALPHA, ALPHA_B, ALPHA_P = 0.05, 0.025, 0.025
SESOI_KL, SESOI_G, MARGIN_E1, MARGIN_E1G = 0.5, 0.20, 0.5, 0.10
FLAG_KL, FLAG_G = 0.18, 0.029                       # "unusually high" within-class label variance (§6.2)
N_LABEL_PAIRS, N_LABEL_CANDS, MIN_MODELS = 24, 40, 4
N_PERM, N_BOOT, BOOT_SEED, MIN_NUMS, MIN_USABLE = 20000, 10000, 20261123, 10, 0.5
BRIDGE_REF = "reference number"

# ================================================================ loading
def load_model(d):
    meta = json.load(open(d / "metadata.json", encoding="utf-8")); sc = {r["job_id"]: r for r in jl(d / "scores.jsonl")}; rows = []
    for r in jl(d / "responses.jsonl"):
        if r["job_id"] not in sc: continue
        s = sc[r["job_id"]]; t = r["response"]
        rows.append(dict(label=r["label"], cls=r["cls"], pair=r["pair_id"], member=r["member"], cond=r["cond"], llr=s["lp_value"] - s["lp_alt"],
                         stratum=stratum4(t, r["value"]), n_nums=len(nums4(t))))
    A = pd.DataFrame(rows)
    S = {r["key"]: r["response"] for r in jl(d / "greedy_single.jsonl")}; B = {r["key"]: r["response"] for r in jl(d / "greedy_batched.jsonl")}
    lc, bc = label_bridge_candidates(meta); G = []
    for label, cls in [(l, c) for c, ls in V7_LABELS.items() for l in ls] + [("secret", "bridge")]:
        conds = ["none"] if cls == "bridge" else ["none", "instr"]
        for cond in conds:
            if f"{label}|{cond}|{(bc if cls == 'bridge' else lc)[0]}" not in S: continue
            if cls != "bridge": G += replay(S, B, label, cls, cond, lc, "label")
            if label in ("secret", BRIDGE_REF) and cond == "none": G += replay(S, B, label, cls, cond, bc, "bridge")
    return dict(meta=meta, A=A, G=pd.DataFrame(G))

def label_bridge_candidates(meta):
    """The notebook records both lists (label cells: first 40; bridge cells: all 80). Older or synthetic metadata: derive them."""
    c = meta["candidates"]; return meta.get("label_candidates", c[:N_LABEL_CANDS]), meta.get("bridge_candidates", c)

def replay(S, B, label, cls, cond, cands, design):
    """v6 replay rule within one cell: groups of identical reference lists among the cell's candidates. Disclosing observed answers
    (stratum 1) are flagged; the primary analysis excludes them, the §4.3 sensitivity analysis keeps them."""
    ref = {v: tuple(nums4(S[f"{label}|{cond}|{v}"])) for v in cands}; groups = {}
    for v, f in ref.items(): groups.setdefault(f, []).append(v)
    out = []
    for v in cands:
        t = B[f"{label}|{cond}|{v}"]; m = groups.get(tuple(nums4(t)), [])
        out.append(dict(label=label, cls=cls, cond=cond, cand=v, score=1 / len(m) if v in m else 0.0, design=design, disclosing=stratum4(t, v) == 1))
    return out

# ================================================================ tables
def units(A, keep):
    """Pair values: mean LLR of the kept answers per label x pair x condition (both directions)."""
    return A[keep].groupby(["label", "cls", "pair", "cond"], as_index=False).llr.mean()

def pooled_table(per_model, rows, cols, value="llr", col="pair"):
    """Complete-pair rule (§4.3): rows (labels, class A first), cols (pairs or candidates). A column missing in ANY model for ANY row of
    the contrast is dropped for all rows and models (reported), so every pooled cell averages the same models and the table is balanced."""
    mats = []
    for U in per_model:
        P = U.pivot_table(index="label", columns=col, values=value, aggfunc="mean").reindex(index=rows, columns=cols); mats.append(P.values)
    S = np.stack(mats); ok = ~np.isnan(S).any((0, 1)); return S[:, :, ok].mean(0), [c for c, k in zip(cols, ok) if not k]

def class_rows(a, b): return V7_LABELS[a] + V7_LABELS[b]

# ================================================================ outcomes
def outcome(rejected, r, margin):
    """§6.1: significance and equivalence are separate flags; equivalence = TOST at 0.05 (90% CI inside +-margin)."""
    equiv = vs.tost_p(r["est"], r["se"], margin, r["df"]) < ALPHA; r["equivalent"] = bool(equiv); r["significant"] = bool(rejected)
    if rejected and equiv: return "detectable but smaller than the smallest effect of interest (also equivalent)"
    if rejected: return "effect"
    return "equivalence" if equiv else "insufficient evidence"

def ci90(r): h = stats.t.ppf(0.95, r["df"]) * r["se"]; return [r["est"] - h, r["est"] + h]

def bridge_unit_diffs(U, pairs):
    W = U[(U.cond == "none") & U.label.isin(["secret", BRIDGE_REF])].pivot_table(index="pair", columns="label", values="llr")
    W = W.reindex(pairs); d = (W["secret"] - W[BRIDGE_REF]) if {"secret", BRIDGE_REF} <= set(W.columns) else pd.Series(dtype=float)
    return d.dropna()

def boot_t_p(d, n_boot=10000, seed=0):
    """Symmetric bootstrap-t p for H0: mean 0 (resamples centred at the sample mean); B1 sensitivity (analysis/v7_validate_bridge.py)."""
    d = np.asarray(d, float)
    if len(d) < 3 or d.std(ddof=1) == 0: return None
    rng = np.random.default_rng(seed); n = len(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(n)); B = d[rng.integers(0, n, (n_boot, n))]
    sb = B.std(1, ddof=1); ok = sb > 0; tb = (B.mean(1)[ok] - d.mean()) / (sb[ok] / np.sqrt(n)); return float(((np.abs(tb) >= abs(t)).sum() + 1) / (ok.sum() + 1))

def t_ci(d, level=0.95):
    d = np.asarray(d, float); se = d.std(ddof=1) / np.sqrt(len(d)); h = stats.t.ppf(0.5 + level / 2, len(d) - 1) * se
    return dict(est=float(d.mean()), se=float(se), lo=float(d.mean() - h), hi=float(d.mean() + h), df=len(d) - 1)

def confirmatory(Zp1, Zp2, Zp3, bridge_d, n_models, seed=0):
    """The eight confirmatory hypotheses (§5.0): B per model (sign-flip over pairs), P1, P2 and the gatekept P3 (crossed), graphical procedure."""
    p1, p2, p3 = vs.crossed(Zp1), vs.crossed(Zp2), vs.crossed(Zp3)
    bres = {m: dict(t_ci(d), p=vs.signflip_p(d, N_PERM, seed + i), n_pairs=len(d)) for i, (m, d) in enumerate(bridge_d.items())}
    p = {**{f"B|{m}": r["p"] for m, r in bres.items()}, "P1": p1["p"], "P2": p2["p"], "P3": p3["p"]}
    w, G = vs.v7_graph([f"B|{m}" for m in bres], ALPHA, ALPHA_B, ALPHA_P)
    rej = vs.graph_test(p, w, G, ALPHA) if n_models >= MIN_MODELS else set()
    for m, r in bres.items(): r["rejected"] = f"B|{m}" in rej; r["outcome"] = outcome(r["rejected"], r, SESOI_KL)
    n_pos = sum(1 for r in bres.values() if r["rejected"] and r["est"] > 0); n_neg = sum(1 for r in bres.values() if r["rejected"] and r["est"] < 0)
    p1.update(rejected="P1" in rej, outcome=outcome("P1" in rej, p1, SESOI_KL), ci90=ci90(p1), flag_high_label_variance=p1["s2_w"] > FLAG_KL)
    p2.update(rejected="P2" in rej, outcome=outcome("P2" in rej, p2, SESOI_G), ci90=ci90(p2), flag_high_label_variance=p2["s2_w"] > FLAG_G)
    p3.update(rejected="P3" in rej, outcome=outcome("P3" in rej, p3, SESOI_KL), ci90=ci90(p3), flag_high_label_variance=p3["s2_w"] > FLAG_KL)
    return dict(B1=dict(per_model=bres, n_secret_higher=n_pos, n_reference_higher=n_neg, most_models_confirmed=n_pos >= 3),
                P1=p1, P2=p2, P3=p3, rejected=sorted(rej), descriptive_only=n_models < MIN_MODELS)

def equivalence(Z, margin, flag):
    r = vs.crossed(Z); r.update(p_tost=vs.tost_p(r["est"], r["se"], margin, r["df"]), ci90=ci90(r), flag_high_label_variance=r["s2_w"] > flag)
    r["outcome"] = outcome(r["p"] < ALPHA, r, margin); return r

def gate3(c, e1, s2_sal_neu):
    """§7. Sensitive vs neutral comes from the confirmatory, gatekept P3; salient vs neutral from E1 and the secondary S2 (p < 0.05),
    so readings that rest on S2 are labelled secondary."""
    p1 = c["P1"]; p3 = c["P3"]; up = lambda r: r["p"] < ALPHA and r["est"] > 0
    p3_up = p3["rejected"] and p3["est"] > 0; p3_down = p3["rejected"] and p3["est"] < 0
    if p1["significant"] and p1["est"] > 0:
        # "raises leakage" needs direct sensitive-vs-neutral evidence (P3), not only sensitive > salient and salient ~ neutral
        if e1["equivalent"] and p3_up: row = "Semantic designation beyond salience (central claim)"
        elif e1["equivalent"] and not p3["rejected"]: row = "Sensitive labels leak more than salient ones, and salient ~ neutral; sensitive vs neutral not resolved"
        elif p3_up: row = "Sensitivity raises leakage; salience may also matter"
        else: row = "Sensitive labels leak more than salient ones; relative to neutral labels not resolved"
    elif p1["significant"] and p1["est"] < 0:
        if up(s2_sal_neu) and p3_down: row = "Salience raises and sensitivity lowers leakage (both against neutral labels; salience part secondary)"
        elif up(s2_sal_neu): row = "Salience, not secrecy, attracts leakage (salient above both sensitive and neutral labels; secondary evidence)"
        elif p3_down: row = "Sensitive designation lowers leakage (suppression; sensitive below salient and neutral labels)"
        else: row = "Salient labels leak more than sensitive ones; whether salience raises or sensitivity lowers leakage is not resolved"
    elif p1["equivalent"] and up(s2_sal_neu): row = "Salience-type effect; sensitivity adds nothing measurable (secondary evidence)"
    elif p1["equivalent"]: row = "Sensitive and salient equivalent; no salience effect shown"
    else: row = "Insufficient evidence on label class; L2 not claimed"
    b = c["B1"]
    if b["most_models_confirmed"]: brow = "B1 confirmed: the v6 'secret' result replicates with fresh 4-digit values"
    elif c.get("B2", {}).get("p", 1) < ALPHA and c["B2"]["est"] > 0: brow = "B1 not confirmed; on average the effect replicates (B2); 'most models' rests on v6"
    elif c.get("B2", {}).get("p", 1) < ALPHA and c["B2"]["est"] < 0: brow = "B2: 'reference number' higher, contradicts v6 at 4 digits"
    else: brow = "Not replicated at 4 digits; the claim rests on v6 alone"
    return dict(family_P=row, family_B=brow, P1_P2_sign_disagree=bool(c["P1"]["rejected"] and c["P2"]["rejected"] and np.sign(c["P1"]["est"]) != np.sign(c["P2"]["est"])))

# ================================================================ full analysis
def analyse(models):
    """models: dict name -> load_model output."""
    names = list(models); M = len(names); meta0 = models[names[0]]["meta"]
    pairs = sorted(set(models[names[0]]["A"].pair))          # pair ids sort in design order (q00..q47; the first 24 = label design)
    label_pairs = meta0.get("label_pairs", pairs[:N_LABEL_PAIRS]); lcands = label_bridge_candidates(meta0)[0]
    use = {m: models[m]["A"].stratum != 1 for m in names}
    U = {m: units(models[m]["A"], use[m]) for m in names}; G0 = {m: models[m]["G"][~models[m]["G"].disclosing] for m in names}
    Gl = {m: G0[m][G0[m].design == "label"] for m in names}; Gb = {m: G0[m][G0[m].design == "bridge"] for m in names}
    def kl_table(a, b, cond, Us=None, ms=None, lp=label_pairs):
        Us = Us or U; ms = ms or names; return pooled_table([Us[m][Us[m].cond == cond] for m in ms], class_rows(a, b), lp)
    def g_table(a, b, cond, ms=None):
        ms = ms or names; return pooled_table([Gl[m][Gl[m].cond == cond] for m in ms], class_rows(a, b), lcands, "score", "cand")
    out = dict(models=names, n_models=M, dropped_columns={})
    Z1, dr1 = kl_table("sensitive", "salient", "none"); Z2, dr2 = g_table("sensitive", "salient", "none")
    out["dropped_columns"].update(P1=dr1, P2=dr2)
    # §4.3: more than 25% of a confirmatory contrast's columns dropped -> flag, and re-run without the model with the most missing cells
    out["missing_flag"] = {}
    for key, dropped, n_cols, fn in [("P1", dr1, len(label_pairs), lambda ms: kl_table("sensitive", "salient", "none", ms=ms)[0]),
                                     ("P2", dr2, len(lcands), lambda ms: g_table("sensitive", "salient", "none", ms=ms)[0])]:
        if len(dropped) > 0.25 * n_cols:
            miss = {m: len((kl_table("sensitive", "salient", "none", ms=[m]) if key == "P1" else g_table("sensitive", "salient", "none", ms=[m]))[1]) for m in names}
            worst = max(miss, key=miss.get); rest = [m for m in names if m != worst]
            out["missing_flag"][key] = dict(share_dropped=len(dropped) / n_cols, model_left_out=worst, without_it=vs.crossed(fn(rest)) if len(rest) else None)
    bd = {m: bridge_unit_diffs(U[m], pairs) for m in names}
    Z3, dr3 = kl_table("sensitive", "neutral", "none"); out["dropped_columns"]["P3"] = dr3
    c = confirmatory(Z1, Z2, Z3, {m: d.values for m, d in bd.items()}, M)
    db = pd.concat([bd[m] for m in names], axis=1).dropna().mean(1)            # complete-pair rule: pairs present in every model
    c["B2"] = dict(t_ci(db), p=vs.signflip_p(db.values, N_PERM, 99), n_pairs=len(db))
    c["B2"]["outcome"] = outcome(c["B2"]["p"] < ALPHA, c["B2"], SESOI_KL)
    # B1 diagnostics (§5, family B): skewness, paired t, symmetric bootstrap-t, literal-free (stratum 4) version; disagreement -> borderline
    U4b = {m: units(models[m]["A"], models[m]["A"].stratum == 4) for m in names}
    for i, m in enumerate(names):
        d = bd[m].values; r = c["B1"]["per_model"][m]; d4 = bridge_unit_diffs(U4b[m], pairs).values
        r["dropped_pairs"] = [p for p in pairs if p not in bd[m].index]           # missing in either bridge cell (§4.3, family B)
        r["diagnostics"] = dict(skewness=float(stats.skew(d)) if len(d) > 2 else None, p_paired_t=float(2 * stats.t.sf(abs(r["est"] / r["se"]), r["df"])) if r["se"] > 0 else None,
                                p_bootstrap_t=boot_t_p(d, seed=500 + i), literal_free=dict(t_ci(d4), p=vs.signflip_p(d4, N_PERM, 600 + i), n_pairs=len(d4)) if len(d4) > 2 else None)
        pb = r["diagnostics"]["p_bootstrap_t"]; lvl = ALPHA_B / len(names)        # the model's initial local level
        r["borderline"] = pb is not None and (r["rejected"] != (pb <= lvl))
    # §4.3 missingness: missing model x label x pair cells by class (P1 and E1 tables, none), and the analysis including disclosing answers
    miss = {}
    for m in names:
        P = U[m][U[m].cond == "none"].pivot_table(index="label", columns="pair", values="llr").reindex(index=[l for ls in V7_LABELS.values() for l in ls], columns=label_pairs)
        for cls, ls in V7_LABELS.items(): miss.setdefault(cls, 0); miss[cls] += int(P.loc[ls].isna().values.sum())
    out["missing_cells_by_class"] = miss
    Ua = {m: units(models[m]["A"], models[m]["A"].stratum > 0) for m in names}
    Gla = {m: models[m]["G"][models[m]["G"].design == "label"] for m in names}
    out["including_disclosing"] = dict(
        P1=vs.crossed(kl_table("sensitive", "salient", "none", Us=Ua)[0]),
        P2=vs.crossed(pooled_table([Gla[m][Gla[m].cond == "none"] for m in names], class_rows("sensitive", "salient"), lcands, "score", "cand")[0]),
        E1=vs.crossed(kl_table("salient", "neutral", "none", Us=Ua)[0]),
        P3=vs.crossed(kl_table("sensitive", "neutral", "none", Us=Ua)[0]),
        B1={m: dict(t_ci(bridge_unit_diffs(Ua[m], pairs).values), p=vs.signflip_p(bridge_unit_diffs(Ua[m], pairs).values, N_PERM, 700 + i)) for i, m in enumerate(names)})
    b3 = {}
    for m in names:
        W = Gb[m].pivot_table(index="cand", columns="label", values="score")
        d = (W["secret"] - W[BRIDGE_REF]).dropna() if {"secret", BRIDGE_REF} <= set(W.columns) else pd.Series(dtype=float)
        b3[m] = dict(est=float(d.mean()) if len(d) else float("nan"), p=vs.signflip_p(d.values, N_PERM, 7) if len(d) > 1 else 1.0, n=len(d))
    hp = sorted(b3, key=lambda k: b3[k]["p"]); stop = False
    for i, m in enumerate(hp):
        b3[m]["holm_rejected"] = (not stop) and b3[m]["p"] <= ALPHA / (M - i); stop = stop or not b3[m]["holm_rejected"]
    c["B3"] = dict(per_model=b3, n_secret_higher=sum(1 for r in b3.values() if r["holm_rejected"] and r["est"] > 0))
    out["confirmatory"] = c
    # equivalence and class secondaries
    e1 = equivalence(kl_table("salient", "neutral", "none")[0], MARGIN_E1, FLAG_KL); e1g = equivalence(g_table("salient", "neutral", "none")[0], MARGIN_E1G, FLAG_G)
    out["E1"], out["E1g"] = e1, e1g
    s2 = {}
    for a, b in [("sensitive", "neutral"), ("salient", "neutral")]:
        for meas, fn, ses, fl in [("kl", kl_table, SESOI_KL, FLAG_KL), ("greedy", g_table, SESOI_G, FLAG_G)]:
            if (a, meas) == ("salient", "greedy"): ses = MARGIN_E1G          # same contrast as E1g: one margin (10 pp), one reading
            r = vs.crossed(fn(a, b, "none")[0]); r.update(outcome=outcome(r["p"] < ALPHA, r, ses), ci90=ci90(r), flag_high_label_variance=r["s2_w"] > fl); s2[f"{a}-{b}|{meas}"] = r
    out["S2"] = s2; out["gate3"] = gate3(c, e1, s2["salient-neutral|kl"])
    # S1 per model
    s1 = {}
    for meas in ("kl", "greedy"):
        ps = {}
        for m in names:
            Z = kl_table("sensitive", "salient", "none", ms=[m])[0] if meas == "kl" else g_table("sensitive", "salient", "none", ms=[m])[0]
            ps[m] = vs.crossed(Z)
        order = sorted(ps, key=lambda k: ps[k]["p"]); stop = False
        for i, m in enumerate(order):
            ps[m]["holm_rejected"] = (not stop) and ps[m]["p"] <= ALPHA / (M - i); stop = stop or not ps[m]["holm_rejected"]
        s1[meas] = ps
    out["S1"] = s1
    # S3 instruction x class
    Ud = {}
    for m in names:
        W = U[m][U[m].pair.isin(label_pairs)].pivot_table(index=["label", "cls", "pair"], columns="cond", values="llr")
        Ud[m] = (W["instr"] - W["none"]).rename("llr").reset_index().assign(cond="diff") if {"instr", "none"} <= set(W.columns) else pd.DataFrame(columns=["label", "cls", "pair", "llr", "cond"])
    out["S3"] = {f"{a}-{b}": vs.crossed(kl_table(a, b, "diff", Us=Ud)[0]) for a, b in [("sensitive", "salient"), ("sensitive", "neutral")]}
    # S4 instruction effect per model (labels as units)
    s4 = {}
    for m in names:
        lm = Ud[m].groupby("label").llr.mean().reindex([l for ls in V7_LABELS.values() for l in ls]).dropna()
        s4[m] = dict(est=float(lm.mean()), p=vs.signflip_p(lm.values, N_PERM, 11), n_labels=len(lm))
    order = sorted(s4, key=lambda k: s4[k]["p"]); stop = False
    for i, m in enumerate(order):
        s4[m]["holm_rejected"] = (not stop) and s4[m]["p"] <= ALPHA / (M - i); stop = stop or not s4[m]["holm_rejected"]
    out["S4"] = dict(per_model=s4, reversal=any(r["holm_rejected"] and r["est"] > 0 for r in s4.values()) and any(r["holm_rejected"] and r["est"] < 0 for r in s4.values()))
    allrows = [l for c in ("sensitive", "salient", "neutral") for l in V7_LABELS[c]]
    out["S6"] = covariate_adjustment(pooled_table([U[m][U[m].cond == "none"] for m in names], allrows, label_pairs)[0].mean(1), allrows)
    # S7 emission, S8 literal-free
    em = []
    for m in names:
        A = models[m]["A"]
        for (cls, cond), g in A.groupby(["cls", "cond"]): em.append(dict(model=m, cls=cls, cond=cond, disclose=float((g.stratum == 1).mean()), exact=float((g.stratum == 2).mean()), n=len(g)))
    out["S7"] = em
    U4 = {m: units(models[m]["A"], models[m]["A"].stratum == 4) for m in names}
    out["S8"] = vs.crossed(kl_table("sensitive", "salient", "none", Us=U4)[0])
    out["robustness"] = robustness(models, names, U, kl_table, g_table, use)
    out["label_level"] = {meas: (lambda Z: dict(perm_p=vs.perm_p(Z.mean(1)), rank_p=vs.perm_p(Z.mean(1), rank=True)))(Z)
                          for meas, Z in [("P1", Z1), ("P2", Z2), ("P3", Z3), ("E1", kl_table("salient", "neutral", "none")[0])]}
    out["per_label"] = per_label(names, U, Gl)
    return out

def covariate_adjustment(y, labels):
    """S6: OLS of the 24 pooled label means (KL, none) on class (salient = reference) and standardized norming covariates."""
    sc = {r["label"]: r for r in json.load(open(RES / "norming_selection_r2.json", encoding="utf-8"))["scores"]}
    y = np.asarray(y, float); cls = ["sensitive"] * 8 + ["salient"] * 8 + ["neutral"] * 8
    out = {}
    for name, covs in [("num_nat_tok", ["num", "nat", "tok"]), ("num_nat_tok_I", ["num", "nat", "tok", "I"])]:
        X = [np.ones(24), np.array([c == "sensitive" for c in cls], float), np.array([c == "neutral" for c in cls], float)]
        for cv in covs:
            v = np.array([sc[l][cv] for l in labels], float); X.append((v - v.mean()) / v.std(ddof=1))
        X = np.column_stack(X); beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta; df = len(y) - X.shape[1]
        cov = res @ res / df * np.linalg.inv(X.T @ X); se = float(np.sqrt(cov[1, 1])); h = stats.t.ppf(0.975, df) * se
        out[name] = dict(sensitive_vs_salient=float(beta[1]), se=se, lo=float(beta[1] - h), hi=float(beta[1] + h), df=df)
    return out

def robustness(models, names, U, kl_table, g_table, use):
    rng = np.random.default_rng(BOOT_SEED); out = {}
    for key, Z in [("P1", kl_table("sensitive", "salient", "none")[0]), ("P2", g_table("sensitive", "salient", "none")[0]),
                   ("P3", kl_table("sensitive", "neutral", "none")[0]), ("E1", kl_table("salient", "neutral", "none")[0])]:
        est = []
        for _ in range(N_BOOT):
            ia = rng.integers(0, 8, 8); ib = 8 + rng.integers(0, 8, 8); jc = rng.integers(0, Z.shape[1], Z.shape[1]); Zb = Z[np.r_[ia, ib]][:, jc]
            est.append(Zb[:8].mean() - Zb[8:].mean())
        out[f"bootstrap|{key}"] = dict(lo=float(np.percentile(est, 2.5)), hi=float(np.percentile(est, 97.5)))
    out["leave_one_model_out"] = {m: dict(P1=vs.crossed(kl_table("sensitive", "salient", "none", ms=[x for x in names if x != m])[0])["est"],
                                          P2=vs.crossed(g_table("sensitive", "salient", "none", ms=[x for x in names if x != m])[0])["est"]) for m in names} if len(names) > 1 else {}
    Z1, Z2 = kl_table("sensitive", "salient", "none")[0], g_table("sensitive", "salient", "none")[0]; lolo = {}
    for i in range(16):
        keep = [j for j in range(16) if j != i]; na = 7 if i < 8 else 8
        lolo[class_rows("sensitive", "salient")[i]] = dict(P1=vs.crossed(Z1[keep], na=na)["est"], P2=vs.crossed(Z2[keep], na=na)["est"])
    out["leave_one_label_out"] = lolo
    # usable-answer rule (v6 A2)
    Uu, excl = {}, []
    for m in names:
        A = models[m]["A"]; usable = use[m] & (A.n_nums >= MIN_NUMS)
        share = usable.groupby([A.cls, A.cond]).mean()
        if (share < MIN_USABLE).any(): excl.append(m)
        Uu[m] = units(A, usable)
    keep = [m for m in names if m not in excl] or names
    out["usable_rule"] = dict(excluded_models=excl, P1=vs.crossed(kl_table("sensitive", "salient", "none", Us=Uu, ms=keep)[0]),
                              P3=vs.crossed(kl_table("sensitive", "neutral", "none", Us=Uu, ms=keep)[0]))
    return out

def per_label(names, U, Gl):
    rows = []
    for m in names:
        for (label, cls, cond), g in U[m].groupby(["label", "cls", "cond"]): rows.append(dict(model=m, label=label, cls=cls, cond=cond, kl=float(g.llr.mean()), n_pairs=len(g)))
        for (label, cls, cond), g in Gl[m].groupby(["label", "cls", "cond"]): rows.append(dict(model=m, label=label, cls=cls, cond=cond, ident=float(g.score.mean()), n_cands=len(g)))
    return rows

# ================================================================ synthetic data
def fake_values(seed):
    rng = np.random.default_rng(seed); seen = set(); pairs = []; cands = []
    def val():
        while True:
            d = rng.choice(10, 4, replace=False)
            if d[0] != 0:
                v = "".join(map(str, d))
                if v not in seen: seen.add(v); return v
    while len(pairs) < 48:
        a = val(); b = val()
        if all(x != y for x, y in zip(a, b)): pairs.append(dict(id=f"q{len(pairs):02d}", a=a, b=b))
    while len(cands) < 80: cands.append(val())
    return pairs, cands

def safe_list(rng, avoid, n=12):
    while True:
        t = "\n".join(str(x) for x in rng.integers(1000, 10000, n))
        if all(stratum4(t, v) != 1 and v not in nums4(t) for v in avoid): return t

def make_fake(root, model_effects, seed=0, disclose_cell=None):
    """model_effects: dict with mu (class means, none), instr (per-class instruction shift), bridge (secret - reference, nats),
    greedy (class identification rates), greedy_bridge (secret - reference, proportion), label_sd, pair_sd, noise_sd."""
    rng = np.random.default_rng(seed); root.mkdir(parents=True, exist_ok=True); e = model_effects
    pairs, cands = fake_values(1234)                         # same values in every model
    lab_eff = {l: np.random.default_rng(int(hashlib.sha256(l.encode()).hexdigest(), 16) % 2**32).normal(0, e.get("label_sd", 0.3)) for ls in V7_LABELS.values() for l in ls}
    pair_eff = {p["id"]: rng.normal(0, e.get("pair_sd", 0.5)) for p in pairs}; resp, sc = [], []
    cells = [(l, c, cond, pairs[:24]) for c, ls in V7_LABELS.items() for l in ls for cond in ("none", "instr")]
    cells += [("secret", "bridge", "none", pairs), (BRIDGE_REF, "neutral", "none", pairs[24:])]
    for label, cls, cond, ps in cells:
        for p in ps:
            for mem in "ab":
                v, alt = (p["a"], p["b"]) if mem == "a" else (p["b"], p["a"]); k = f"{label}|{p['id']}|{mem}|{cond}"
                if cls == "bridge": base = e["mu"]["neutral"] + lab_eff[BRIDGE_REF] + e["bridge"]          # secret = reference number + bridge effect
                else: base = e["mu"][cls] + lab_eff[label] + (e["instr"][cls] if cond == "instr" else 0.0)
                llr = base + pair_eff[p["id"]] + rng.normal(0, e.get("noise_sd", 1.0))
                t = "\n".join([v] * 12) if (label, p["id"]) in (disclose_cell or []) else safe_list(rng, [v, alt])
                resp.append(dict(job_id=k, label=label, cls=cls, pair_id=p["id"], member=mem, cond=cond, value=v, alt=alt, response=t))
                sc.append(dict(job_id=k, lp_value=-50.0 + llr, lp_alt=-50.0))
    gs, gb = [], []
    gcells = [(l, c, cond, cands[:40]) for c, ls in V7_LABELS.items() for l in ls for cond in ("none", "instr")]
    gcells += [("secret", "bridge", "none", cands), (BRIDGE_REF, "neutral", "none", cands[40:])]
    for label, cls, cond, cs in gcells:
        rate = e["greedy"]["neutral"] + e["greedy_bridge"] if cls == "bridge" else e["greedy"][cls]
        for v in cs:
            key = f"{label}|{cond}|{v}"; r2 = np.random.default_rng(int(hashlib.sha256((key + str(seed)).encode()).hexdigest(), 16) % 2**32)
            t = safe_list(r2, [v], 6); gs.append(dict(key=key, response=t)); gb.append(dict(key=key, response=t if r2.random() < rate else safe_list(r2, [v], 6)))
    meta = dict(model_id="fake", prompt_hash="x", run_complete=True, labels=[(l, c) for c, ls in V7_LABELS.items() for l in ls] + [("secret", "bridge")],
                pairs=pairs, candidates=cands)
    for name, rows in [("responses.jsonl", resp), ("scores.jsonl", sc), ("greedy_single.jsonl", gs), ("greedy_batched.jsonl", gb)]:
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

def selftest():
    ok = []
    def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
    # unit: graph reproduces Holm within one family; crossed equals the validated implementation
    rng = np.random.default_rng(1); fam = [f"h{i}" for i in range(5)]; good = True
    for _ in range(300):
        q = {h: float(x) for h, x in zip(fam, rng.random(5) ** 3)}; w, G = vs.two_family_graph(fam, ["x"], 0.05, 0.0)
        rej = vs.graph_test({**q, "x": 1.0}, w, G); srt = sorted(fam, key=lambda h: q[h]); holm = set()
        for i, h in enumerate(srt):
            if q[h] <= 0.05 / (5 - i): holm.add(h)
            else: break
        good &= rej == holm
    check("graph_test equals Holm within one family (300 random cases)", good)
    w, G = vs.v7_graph(fam); base = {h: 1.0 for h in fam}
    gate_ok = (vs.graph_test({**base, "P1": 1.0, "P2": 1.0, "P3": 1e-12}, w, G) == set()
               and vs.graph_test({**base, "P1": 0.01, "P2": 1.0, "P3": 0.012}, w, G) == {"P1", "P3"}
               and vs.graph_test({**{h: 0.0099 for h in fam}, "P1": 0, "P2": 0, "P3": 0}, w, G) == set(fam) | {"P1", "P2", "P3"})
    check("v7_graph: P3 is tested only after P1 is rejected; P passes its alpha to B only after P1-P3 are all rejected", gate_ok)
    import importlib.util
    spec = importlib.util.spec_from_file_location("vt", ROOT / "analysis" / "v7_validate_test.py"); vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
    Z = rng.normal(0, 1, (16, 24)); a = vs.crossed(Z); b = vt.crossed_ci(Z, 0.95)
    o = [outcome(rj, dict(est=e, se=0.1, df=14), 0.5) for rj, e in [(True, 1.0), (True, 0.25), (False, 0.0), (False, 0.4)]]
    check("outcome(): effect / detectable-but-small (also equivalent) / equivalence / insufficient evidence", o[0] == "effect" and o[1].startswith("detectable")
          and o[2] == "equivalence" and o[3] == "insufficient evidence", str(o))
    check("crossed() equals the validated crossed_ci", abs(a["lo"] - b[0]) < 1e-10 and abs(a["hi"] - b[1]) < 1e-10 and abs(a["p"] - b[2]) < 1e-10)
    # error rate of the confirmatory layer under the global null (in-memory tables)
    rej_any = 0; R = 300
    for r in range(R):
        Z1 = rng.normal(0, 0.4, (16, 1)) + rng.normal(0, 1, (16, 24)) / 3; Z2 = rng.random((16, 40)) < 0.6
        Z3 = rng.normal(0, 0.4, (16, 1)) + rng.normal(0, 1, (16, 24)) / 3
        bd = {m: rng.normal(0, 1.4, 48) for m in MODELS}; c = confirmatory(Z1, Z2.astype(float), Z3, bd, 5, seed=r)
        rej_any += bool(c["rejected"])
    check(f"global null: familywise error of the eight confirmatory tests <= 0.07 over {R} replications", rej_any / R <= 0.07, f"{rej_any / R:.3f}")
    base = dict(mu=dict(sensitive=1.0, salient=1.0, neutral=1.0), instr=dict(sensitive=0.0, salient=0.0, neutral=0.0), bridge=0.0,
                greedy=dict(sensitive=0.6, salient=0.6, neutral=0.6), greedy_bridge=0.0, label_sd=0.15, pair_sd=0.4, noise_sd=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def run(tag, effects_by_model, disclose=None):
            ms = {}
            for i, (m, e) in enumerate(effects_by_model.items()):
                d = tmp / tag / m; make_fake(d, e, seed=100 + i, disclose_cell=(disclose or {}).get(m)); ms[m] = load_model(d)
            return analyse(ms)
        planted = {}
        for i, m in enumerate(MODELS):
            e = json.loads(json.dumps(base)); e["mu"]["sensitive"] = 2.0; e["greedy"]["sensitive"] = 0.85; e["bridge"] = 1.5 if i < 4 else 0.0
            e["greedy_bridge"] = 0.25 if i < 4 else 0.0; e["instr"] = {c: (0.9 if i < 2 else -0.9) for c in V7_LABELS}; planted[m] = e
        r = run("planted", planted); c = r["confirmatory"]
        check("planted: B1 'most models' confirmed (4 models planted), the null model not rejected", c["B1"]["most_models_confirmed"] and not c["B1"]["per_model"][MODELS[4]]["rejected"],
              f"{c['B1']['n_secret_higher']} secret-higher")
        check("planted: P3 (gatekept sensitive - neutral) rejected, sensitive higher", c["P3"]["rejected"] and c["P3"]["est"] > 0, f"P3 {c['P3']['est']:+.2f} p {c['P3']['p']:.2g}")
        check("planted: P1 and P2 rejected, sensitive higher", c["P1"]["rejected"] and c["P2"]["rejected"] and c["P1"]["est"] > 0 and c["P2"]["est"] > 0,
              f"P1 {c['P1']['est']:+.2f} p {c['P1']['p']:.2g}; P2 {c['P2']['est']:+.2f} p {c['P2']['p']:.2g}")
        check("planted: E1 equivalence (salient = neutral)", r["E1"]["outcome"] == "equivalence", f"{r['E1']['est']:+.2f} 90% CI {np.round(r['E1']['ci90'], 2)}")
        check("planted: Gate 3 reads the central claim; family B reads replicated", r["gate3"]["family_P"].startswith("Semantic designation") and r["gate3"]["family_B"].startswith("B1 confirmed"))
        check("planted: S4 instruction reversal (2 models up, 3 down)", r["S4"]["reversal"])
        check("planted: B2 and B3 positive", r["confirmatory"]["B2"]["est"] > 0 and r["confirmatory"]["B3"]["n_secret_higher"] >= 3, f"B3 {r['confirmatory']['B3']['n_secret_higher']}")
        check("planted: secondaries, covariate adjustment, robustness, label-level tests present",
              all(k in r for k in ("S1", "S2", "S3", "S6", "S7", "S8", "robustness", "label_level")) and r["robustness"]["bootstrap|P1"]["lo"] > 0)
        dg = [v["diagnostics"] for v in c["B1"]["per_model"].values()]
        check("planted: B1 diagnostics (skewness, paired t, bootstrap-t, literal-free), dropped pairs and the including-disclosing B1 present; planted models agree (no borderline)",
              all(x["p_bootstrap_t"] is not None and x["literal_free"] is not None for x in dg) and "including_disclosing" in r
              and all(v["dropped_pairs"] == [] for v in c["B1"]["per_model"].values()) and len(r["including_disclosing"]["B1"]) == 5
              and not any(v["borderline"] for v in c["B1"]["per_model"].values()), str([round(x["p_bootstrap_t"], 4) for x in dg]))
        c2 =json.loads(json.dumps(c, default=str)); c2["P1"].update(significant=True, est=0.8)
        c2["P3"].update(rejected=False, est=0.3)
        g = gate3(c2, dict(equivalent=True), dict(p=0.5, est=0.0))
        check("P3 in every robustness path (including-disclosing, bootstrap, label-level, usable rule)",
              all(k in r["including_disclosing"] for k in ("P1", "P2", "P3", "E1")) and "bootstrap|P3" in r["robustness"] and "P3" in r["label_level"] and "P3" in r["robustness"]["usable_rule"])
        x = dict(r["S2"]["salient-neutral|greedy"]); same = outcome(x["p"] < ALPHA, dict(x), MARGIN_E1G) == r["S2"]["salient-neutral|greedy"]["outcome"]
        check("S2 greedy salient - neutral read with E1g's margin (10 pp), so it cannot disagree with E1g", same and r["S2"]["salient-neutral|greedy"]["equivalent"] == r["E1g"]["equivalent"])
        c3 = json.loads(json.dumps(c2)); c3["P3"].update(rejected=True, est=-0.4)
        g3 = gate3(c3, dict(equivalent=True), dict(p=0.5, est=0.0))
        check("Gate 3: sensitive > salient, salient ~ neutral, P3 rejected NEGATIVE -> the 'otherwise' row, not 'P3 not rejected'", g3["family_P"].startswith("Sensitive labels leak more than salient ones; relative"), g3["family_P"])
        check("Gate 3: sensitive > salient and salient ~ neutral but P3 (sensitive - neutral) not rejected -> NOT the central claim", g["family_P"].startswith("Sensitive labels leak more than salient ones, and salient"), g["family_P"])
        rev = {m: dict(e, mu=dict(sensitive=1.0, salient=2.2, neutral=1.0), greedy=dict(sensitive=0.6, salient=0.85, neutral=0.6)) for m, e in planted.items()}
        r = run("reversed", rev)
        check("reversed (salient above sensitive AND neutral): P1 rejected, salient higher; Gate 3 reads 'salience, not secrecy'", r["confirmatory"]["P1"]["rejected"] and r["confirmatory"]["P1"]["est"] < 0
              and r["gate3"]["family_P"].startswith("Salience, not secrecy"), r["gate3"]["family_P"])
        sup = {m: dict(e, mu=dict(sensitive=0.0, salient=1.0, neutral=1.0), greedy=dict(sensitive=0.4, salient=0.6, neutral=0.6)) for m, e in planted.items()}
        r = run("suppression", sup)
        check("suppression (sensitive below salient = neutral): Gate 3 reads 'sensitive designation lowers leakage', not 'salience'",
              r["gate3"]["family_P"].startswith("Sensitive designation lowers"), r["gate3"]["family_P"])
        r = run("null", {m: base for m in MODELS})
        check("null: no confirmatory rejection; B reads not replicated; P reads insufficient evidence or equivalence",
              not r["confirmatory"]["rejected"] and r["gate3"]["family_B"].startswith("Not replicated") and r["confirmatory"]["P1"]["outcome"] in ("insufficient evidence", "equivalence"),
              f"rejected {r['confirmatory']['rejected']}; P1 {r['confirmatory']['P1']['outcome']}")
        r = run("missing", {m: base for m in MODELS[:4]}, disclose={MODELS[1]: [("access code", "q03")]})
        check("edge: a label x pair missing in ONE model drops that pair for the whole contrast (complete-pair rule, reported); 4 models still confirmatory",
              r["dropped_columns"]["P1"] == ["q03"] and not r["confirmatory"]["descriptive_only"], str(r["dropped_columns"]))
        check("edge: missing cells reported by class; the including-disclosing analysis keeps the pair", r["missing_cells_by_class"] == {"sensitive": 1, "salient": 0, "neutral": 0}
              and r["including_disclosing"]["P1"]["est"] is not None, str(r["missing_cells_by_class"]))
        r = run("manymissing", {m: base for m in MODELS[:4]}, disclose={MODELS[2]: [("card number", f"q{i:02d}") for i in range(8)]})
        check("edge: > 25% of P1's pairs missing (all in one model) -> flagged, re-run without that model", r["missing_flag"].get("P1", {}).get("model_left_out") == MODELS[2]
              and r["missing_flag"]["P1"]["without_it"] is not None, str({k: (v["share_dropped"], v["model_left_out"]) for k, v in r["missing_flag"].items()}))
        r = run("three", {m: base for m in MODELS[:3]})
        check("edge: fewer than 4 models -> descriptive only, nothing rejected", r["confirmatory"]["descriptive_only"] and not r["confirmatory"]["rejected"])
    print(f"{sum(ok)}/{len(ok)} checks passed"); return all(ok)

if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    models = {}
    for m in MODELS:
        d = RES / f"v7_{m}"
        if (d / "metadata.json").exists() and json.load(open(d / "metadata.json", encoding="utf-8")).get("run_complete"): models[m] = load_model(d)
    assert models, "no complete results/v7_<model>/ folders"
    out = analyse(models)
    def js(o):
        if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)): return [js(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o
    (RES / "v7_analysis.json").write_text(json.dumps(js(out), indent=1), encoding="utf-8")
    c = out["confirmatory"]; print("rejected:", c["rejected"]); print("Gate 3:", out["gate3"])
