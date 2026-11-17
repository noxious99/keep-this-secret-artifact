"""Pre-registered analysis of the attacker study (preregistration_attack.md §4-§7). Written and tested on synthetic data before
any attacker-study data exist.

Usage:
  python analysis/attack_analyze.py              real data (results/attack_<model>/ + stored greedy runs) -> results/attack_analysis.json
  python analysis/attack_analyze.py --selftest   synthetic planted / copy-only / null data in a temporary directory

Files per model (results/attack_<tag>/), written by the attacker-study notebook:
  armA_sampled.jsonl         {"role": "ref"|"target", "cond", "T", "value", "idx", "response"}
  armA_greedy_single.jsonl   {"key": "<cond>|<value>", "response"}   only cells missing from the stored greedy runs (Mistral ref)
  armA_greedy_batched.jsonl  same, batched calls
  armA_ceiling.jsonl         {"cond", "T", "value", "idx", "scores": {candidate: log-prob of the target under that candidate}}
  armA_decoder.jsonl         optional: {"cond", "T", "value", "idx", "probs": [[10 floats] x 2]}
  armB_sampled.jsonl         {"cond", "value", "split": "train"|"val"|"test", "idx", "response"}
  armB_decoder.jsonl         {"decoder": "direct"|"shuffled", "train_cond", "cond", "seed", "n_train", "value", "idx", "probs": [[10] x 4]}
Stored greedy answers (reused, §4): greedy_single.jsonl / greedy_batched.jsonl of the v6 run (v5b for Mistral), key "<instr>|<value>"."""
import json, math, re, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r0_common import RES, nums_clean, structural, verbal, digit_disc, stratum as stratum2, holm, signflip, jsonable
sys.stdout.reconfigure(encoding="utf-8")

ALPHA, MARGIN, N_PERM = 0.05, 0.03, 20000
CANDS = [str(n) for n in range(10, 100) if str(n)[0] != str(n)[1] and n != 20]          # the 80 Arm A candidates
NUMS90 = [f"{n:02d}" for n in range(10, 100)]; N90 = {x: i for i, x in enumerate(NUMS90)}
MODELS = {"Phi-4": ("phi-4", "pilot_v6_phi-4"), "Falcon3-10B": ("falcon3-10b-instruct", "pilot_v6_falcon3-10b-instruct"),
          "Granite-3.3-8B": ("granite-33-8b-instruct", "pilot_v6_granite-33-8b-instruct"),
          "Llama-3.1-8B": ("llama-31-8b-instruct", "pilot_v6_llama-31-8b-instruct"),
          "Mistral-Small-24B": ("mistral-small-24b-instruct-2501", "pilot_v5b_mistral-small-24b-instruct-2501")}
def jl(p): p = Path(p); return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

# ================================================================ parsing and strata
WORD = re.compile(r"[A-Za-z]{3,}")
def nums4(t):
    out = []
    for l in t.splitlines():
        if WORD.search(l): continue
        out += re.findall(r"(?<!\d)\d{4}(?!\d)", re.sub(r"^\s*\d+[.)]\s+", "", l))
    return out
def stratum4(t, v):
    """§5.4: 1 disclosure, 2 exact, 3 partial (inside a longer run, or two consecutive digit runs concatenating to v), 4 none."""
    if structural(t, v) or verbal(t, v) or digit_disc(t, v): return 1
    if v in nums4(t): return 2
    runs = re.findall(r"\d+", t)
    if any(len(r) > 4 and v in r for r in runs) or any(runs[i] + runs[i + 1] == v for i in range(len(runs) - 1)): return 3
    return 4

# ================================================================ tests
def perm_above_chance(S, seed=0):
    """S[i, j] = mean score of value i's answers if the truth were value j; one-sided label permutation (values as units)."""
    obs = float(np.mean(np.diag(S))); n = len(S); rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((N_PERM, n)), axis=1); null = S[np.arange(n)[None, :], perms].mean(1)
    return obs, float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))
def paired(x, one_sided=True):
    x = pd.Series(x).dropna()
    if len(x) < 2: return dict(mean=float("nan"), p=float("nan"), n=int(len(x)))
    p = float(signflip(x, one_sided=one_sided)) if np.any(x != 0) else 1.0
    return dict(mean=float(x.mean()), p=p, n=int(len(x)))

# ================================================================ Arm A (closed set)
def best_set(scores, tol=1e-9):
    m = max(scores.values()); return [c for c, s in scores.items() if s >= m - tol]
def frac(best, truth): return 1.0 / len(best) if truth in best else 0.0
def profiles(refs_by_cand, n_refs):
    P = {}
    for c in CANDS:
        cnt = np.full(90, 0.5)
        for r in refs_by_cand.get(c, [])[:n_refs]:
            for x in nums_clean(r):
                if x in N90: cnt[N90[x]] += 1
        P[c] = np.log(cnt / cnt.sum())
    return P
def counts90(t):
    v = np.zeros(90)
    for x in nums_clean(t):
        if x in N90: v[N90[x]] += 1
    return v
def sim_positional(o, r):
    if not o and not r: return 0.0
    return sum(1 for i in range(min(len(o), len(r))) if o[i] == r[i]) / max(len(o), len(r))

def arm_a_cell(rows, cond, T, n_refs=20, k=1, attacker="freq", ceiling=None, decoder=None):
    """Per-candidate identification for one (cond, T). Returns dict cand -> score, and the S matrix for the permutation test."""
    refs = {}; tg = {}
    for r in rows:
        if r["cond"] != cond or abs(r["T"] - T) > 1e-9: continue
        (refs if r["role"] == "ref" else tg).setdefault(r["value"], []).append(r)
    for d in (refs, tg):
        for c in d: d[c] = [x["response"] for x in sorted(d[c], key=lambda x: x["idx"])]
    P = profiles(refs, n_refs) if attacker == "freq" else None
    if attacker == "nearest":                                                      # reference matrix: padded number indices (-1 never matches)
        seqs, RIDX = [], {}
        for d in CANDS:
            a = len(seqs); seqs += [[int(x) for x in nums_clean(r)] for r in refs.get(d, [])[:n_refs]]; RIDX[d] = (a, len(seqs))
        Lm = max([len(s) for s in seqs] + [1]); RA = np.full((len(seqs), max(Lm, 40)), -1); RL = np.array([len(s) for s in seqs])
        for i, s in enumerate(seqs): RA[i, :len(s)] = s
    cand_scores = {}; S = np.zeros((len(CANDS), len(CANDS))); ci = {c: i for i, c in enumerate(CANDS)}
    for c in CANDS:
        ans = [t for t in tg.get(c, []) if stratum2(t, c, nums_clean(t)) != 1]
        if not ans: continue
        groups = [ans[i:i + k] for i in range(0, len(ans), k)] if k > 1 else [[a] for a in ans]
        if k > 1: groups = [g for g in groups if len(g) == k][:1] or [ans]
        rowS = np.zeros(len(CANDS)); vals = []
        for g in groups:
            if attacker == "freq":
                cv = sum(counts90(t) for t in g); sc = {d: float(cv @ P[d]) for d in CANDS}
            elif attacker == "nearest":                                          # vectorized sim_positional, max over each candidate's references
                sc = {d: 0.0 for d in CANDS}
                for t in g:
                    o = [int(x) for x in nums_clean(t)]; op = np.full(RA.shape[1], -2); op[:len(o)] = o[:RA.shape[1]]
                    sim = (RA == op).sum(1) / np.maximum(np.maximum(RL, len(o)), 1)
                    for d, (a, b) in RIDX.items(): sc[d] += float(sim[a:b].max()) if b > a else 0.0
            elif attacker == "ceiling":
                sc = {d: sum(ceiling[(cond, T, c, t)][d] for t in g) for d in CANDS}
            elif attacker == "decoder":
                sc = {d: sum(math.log(max(decoder[(cond, T, c, t)][0][int(d[0])], 1e-12)) + math.log(max(decoder[(cond, T, c, t)][1][int(d[1])], 1e-12)) for t in g) for d in CANDS}
            b = best_set(sc); vals.append(frac(b, c))
            for d in b: rowS[ci[d]] += 1.0 / len(b)
        cand_scores[c] = float(np.mean(vals)); S[ci[c]] = rowS / len(groups)
    keep = [ci[c] for c in cand_scores]
    return cand_scores, S[np.ix_(keep, keep)]

def replay_scores(single, batched, cond):
    """Pre-registered replay rule (v5b/v6): references = single greedy answers, observed = batched; disclosing candidates excluded."""
    ref = {c: tuple(nums_clean(single[f"{cond}|{c}"])) for c in CANDS if f"{cond}|{c}" in single}
    groups = {}
    for c, f in ref.items(): groups.setdefault(f, []).append(c)
    out = {}
    for c in ref:
        if f"{cond}|{c}" not in batched: continue
        t = batched[f"{cond}|{c}"]
        if structural(t, c) or verbal(t, c) or digit_disc(t, c): continue
        m = groups.get(tuple(nums_clean(t)), []); out[c] = 1.0 / len(m) if c in m else 0.0
    return out

def analyse_arm_a(d, stored_dir):
    rows = jl(d / "armA_sampled.jsonl"); R = {}
    if not rows: return dict(missing=True)
    ceil = {}
    for r in jl(d / "armA_ceiling.jsonl"):
        resp = next(x["response"] for x in rows if x["role"] == "target" and x["cond"] == r["cond"] and abs(x["T"] - r["T"]) < 1e-9 and x["value"] == r["value"] and x["idx"] == r["idx"])
        ceil[(r["cond"], r["T"], r["value"], resp)] = r["scores"]
    dec = {}
    for r in jl(d / "armA_decoder.jsonl"):
        resp = next(x["response"] for x in rows if x["role"] == "target" and x["cond"] == r["cond"] and abs(x["T"] - r["T"]) < 1e-9 and x["value"] == r["value"] and x["idx"] == r["idx"])
        dec[(r["cond"], r["T"], r["value"], resp)] = r["probs"]
    for cond in ["none", "ord", "ref"]:
        for T in [1.0, 0.3]:
            cs, S = arm_a_cell(rows, cond, T)
            if not cs: continue
            obs, p = perm_above_chance(S)
            cell = dict(primary_freq=dict(acc=100 * obs, p_perm=p, n=len(cs)))
            cell["nearest"] = 100 * float(np.mean(list(arm_a_cell(rows, cond, T, attacker="nearest")[0].values())))
            cell["refs_curve"] = {n: 100 * float(np.mean(list(arm_a_cell(rows, cond, T, n_refs=n)[0].values()))) for n in (1, 5, 20)}
            cell["k3"] = 100 * float(np.mean(list(arm_a_cell(rows, cond, T, k=3)[0].values())))
            if dec: cell["decoder"] = 100 * float(np.mean(list(arm_a_cell(rows, cond, T, attacker="decoder", decoder=dec)[0].values())))
            if ceil and T == 1.0 and cond in ("none", "ord"):
                cc, _ = arm_a_cell(rows, cond, T, attacker="ceiling", ceiling=ceil); a, cl = obs, float(np.mean(list(cc.values())))
                cell["ceiling"] = 100 * cl; cell["normalized_extraction"] = (a - 1 / 80) / (cl - 1 / 80) if cl > 1 / 80 else float("nan")
                cell["_ceiling_per_cand"] = cc
            R[f"{cond}|T{T}"] = cell
    # greedy replay: stored run plus new cells
    single = {r["key"]: r["response"] for r in jl(stored_dir / "greedy_single.jsonl")}; single.update({r["key"]: r["response"] for r in jl(d / "armA_greedy_single.jsonl")})
    batched = {r["key"]: r["response"] for r in jl(stored_dir / "greedy_batched.jsonl")}; batched.update({r["key"]: r["response"] for r in jl(d / "armA_greedy_batched.jsonl")})
    rep = {c: replay_scores(single, batched, c) for c in ["none", "ord", "ref"]}
    R["greedy_replay"] = {c: 100 * float(np.mean(list(v.values()))) if v else float("nan") for c, v in rep.items()}
    # H-A3: [ceiling(ord) - ceiling(none)] at T = 1 minus [replay(ord) - replay(none)] at greedy, per candidate (two-sided)
    co, cn = R.get("ord|T1.0", {}).get("_ceiling_per_cand"), R.get("none|T1.0", {}).get("_ceiling_per_cand")
    if co and cn:
        common = [c for c in CANDS if c in co and c in cn and c in rep["ord"] and c in rep["none"]]
        x = [(co[c] - cn[c]) - (rep["ord"][c] - rep["none"][c]) for c in common]
        R["H_A3"] = paired(x, one_sided=False)
    for v in R.values():
        if isinstance(v, dict): v.pop("_ceiling_per_cand", None)
    return R

# ================================================================ Arm B (open set)
def logpost_freq(train_rows):
    """Fairoze App. E (2): per-position digit-frequency profiles P_j(x | secret digit d), Laplace-smoothed."""
    M = np.ones((4, 10, 10))
    for r in train_rows:
        if stratum4(r["response"], r["value"]) == 1: continue
        for n in nums4(r["response"]):
            for j in range(4): M[j, int(r["value"][j]), int(n[j])] += 1
    return np.log(M / M.sum(2, keepdims=True))
def freq_logprobs(LP, t):
    c = np.zeros((4, 10))
    for n in nums4(t):
        for j in range(4): c[j, int(n[j])] += 1
    lp = np.einsum("jdx,jx->jd", LP, c); return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)

def value_metrics(lp_by_value, truth_digits, k=1):
    """lp_by_value: value -> list of (4, 10) log-prob arrays (answers in idx order). Per value: per-position accuracy at k
    (mean over the value's answer groups), exact match, channel-entropy contribution."""
    out = {}
    for v, L in lp_by_value.items():
        if not L: continue
        groups = [L[i:i + k] for i in range(0, len(L), k)] if k > 1 else [[x] for x in L]
        if k > 1: groups = [g for g in groups if len(g) == k][:1] or [L]
        acc, em, ce = [], [], []
        for g in groups:
            s = np.sum(g, axis=0); s = s - np.logaddexp.reduce(s, axis=1, keepdims=True); pred = s.argmax(1); tr = truth_digits[v]
            acc.append(float(np.mean(pred == tr))); em.append(float(np.all(pred == tr)))
            ce.append(float(-np.sum(s[np.arange(4), tr]) / math.log(2)))
        out[v] = dict(acc=float(np.mean(acc)), exact=float(np.mean(em)), ce_bits=float(np.mean(ce)))
    return out

def analyse_arm_b(d):
    rows = jl(d / "armB_sampled.jsonl"); preds = jl(d / "armB_decoder.jsonl")
    if not rows: return dict(missing=True)
    R = {}
    for cond in ["ord", "ref"]:
        cr = [r for r in rows if r["cond"] == cond]
        train = [r for r in cr if r["split"] == "train"]; test = [r for r in cr if r["split"] == "test"]
        if not test: continue
        for r in test: r["_s"] = stratum4(r["response"], r["value"])
        tv = sorted({r["value"] for r in test}); truth = {v: np.array([int(ch) for ch in v]) for v in tv}
        maj = np.array([pd.Series([int(r["value"][j]) for r in train]).value_counts().idxmax() for j in range(4)])
        base = {v: float(np.mean(maj == truth[v])) for v in tv}
        LP = logpost_freq(train)
        def by_value(lp_fn, keep):
            out = {v: [] for v in tv}
            for r in sorted(test, key=lambda r: (r["value"], r["idx"])):
                if keep(r): out[r["value"]].append(lp_fn(r))
            return out
        pm = {}
        for p in preds:
            if p["cond"] == cond: pm[(p["decoder"], p["train_cond"], p["seed"], p["n_train"], p["value"], p["idx"])] = np.log(np.clip(np.array(p["probs"]), 1e-12, 1))
        def dec_fn(decoder="direct", train_cond=cond, seed=0, n_train=500):
            return lambda r: pm[(decoder, train_cond, seed, n_train, r["value"], r["idx"])]
        has = lambda *key: any(k[:4] == key for k in pm)
        s24 = lambda r: r["_s"] in (2, 3, 4); s4 = lambda r: r["_s"] == 4
        cell = dict(strata={str(s): int(sum(r["_s"] == s for r in test)) for s in (1, 2, 3, 4)}, majority=float(np.mean(list(base.values()))))
        fm = value_metrics(by_value(lambda r: freq_logprobs(LP, r["response"]), s24), truth)
        cell["freq_bayes"] = dict(acc=float(np.mean([m["acc"] for m in fm.values()])))
        if has("direct", cond, 0, 500):
            dm = value_metrics(by_value(dec_fn(), s24), truth)
            cell["decoder"] = dict(acc=float(np.mean([m["acc"] for m in dm.values()])), exact=float(np.mean([m["exact"] for m in dm.values()])),
                                   channel_entropy_bits=float(np.mean([m["ce_bits"] for m in dm.values()])))
            cell["vs_majority"] = paired([dm[v]["acc"] - base[v] for v in dm])
            cell["vs_freq_bayes"] = paired([dm[v]["acc"] - fm[v]["acc"] for v in dm if v in fm])
            lf = value_metrics(by_value(dec_fn(), s4), truth)
            cell["literal_free"] = dict(vs_majority=paired([lf[v]["acc"] - base[v] for v in lf]), n_values=len(lf), dropped=len(tv) - len(lf))
            cell["k_curve"] = {k: float(np.mean([m["acc"] for m in value_metrics(by_value(dec_fn(), s24), truth, k=k).values()])) for k in (1, 3, 10)}
            cell["learning_curve"] = {n: float(np.mean([m["acc"] for m in value_metrics(by_value(dec_fn(n_train=n), s24), truth).values()]))
                                      for n in (50, 100, 200, 500) if has("direct", cond, 0, n)}
            cell["seeds"] = {s: float(np.mean([m["acc"] for m in value_metrics(by_value(dec_fn(seed=s), s24), truth).values()])) for s in (1, 2) if has("direct", cond, s, 500)}
            other = "ref" if cond == "ord" else "ord"
            if has("direct", other, 0, 500):
                cell["transfer_from_" + other] = float(np.mean([m["acc"] for m in value_metrics(by_value(dec_fn(train_cond=other), s24), truth).values()]))
        if has("shuffled", cond, 0, 500):
            sm = value_metrics(by_value(dec_fn(decoder="shuffled"), s24), truth)
            cell["shuffled_control"] = paired([sm[v]["acc"] - base[v] for v in sm])
            # Amendment A1: invalid only if the shuffled control beats the majority baseline significantly AND by the practical margin
            cell["valid"] = not (cell["shuffled_control"]["p"] < ALPHA and cell["shuffled_control"]["mean"] >= MARGIN)
        R[cond] = cell
    return R

# ================================================================ families, decision
def families(per_model):
    out = {}
    def fam(name, getter):
        names = [m for m in per_model if getter(per_model[m]) is not None]
        ps = [getter(per_model[m])["p"] for m in names]; adj = holm(ps) if ps else []
        out[name] = {m: dict(**getter(per_model[m]), p_holm=pa) for m, pa in zip(names, adj)}
    valid = lambda R, c: R.get("arm_b", {}).get(c, {}).get("valid", True)
    fam("H_A1", lambda R: R.get("arm_b", {}).get("ord", {}).get("vs_majority") if valid(R, "ord") else None)
    fam("H_A1b", lambda R: R.get("arm_b", {}).get("ord", {}).get("vs_freq_bayes") if valid(R, "ord") else None)
    fam("H_A2", lambda R: R.get("arm_b", {}).get("ref", {}).get("vs_majority") if valid(R, "ref") else None)
    fam("H_A1_LF", lambda R: R.get("arm_b", {}).get("ord", {}).get("literal_free", {}).get("vs_majority") if valid(R, "ord") else None)
    fam("H_A3", lambda R: R.get("arm_a", {}).get("H_A3"))
    sig = lambda f, m: out[f].get(m, {}).get("p_holm", 1) < ALPHA
    strong = [m for m in out["H_A1"] if sig("H_A1", m) and out["H_A1"][m]["mean"] >= MARGIN]
    grade = "strong" if len(strong) >= 3 else "partial" if strong else "none"
    lf = [m for m in out["H_A1_LF"] if sig("H_A1_LF", m)]
    reading = ("practical recovery from sampled answers, not reducible to copying" if grade == "strong" and len(lf) >= 2 else
               "recovery runs through the value appearing in the lists" if grade == "strong" else
               "model-dependent attack" if grade == "partial" else "no practical recovery under these conditions")
    out["decision"] = dict(grade=grade, models_meeting_H_A1_with_margin=strong, literal_free_significant=lf, reading=reading,
                           H_A2_significant=[m for m in out["H_A2"] if sig("H_A2", m)], H_A3_significant=[m for m in out["H_A3"] if sig("H_A3", m)],
                           invalid_decoders=[m for m in per_model for c in ("ord", "ref") if not valid(per_model[m], c)])
    return out

def run(model_dirs):
    per = {}
    for name, (d, stored) in model_dirs.items():
        per[name] = dict(arm_a=analyse_arm_a(Path(d), Path(stored)), arm_b=analyse_arm_b(Path(d)))
    return dict(per_model=per, families=families(per))

# ================================================================ synthetic data (self-test)
def make_fake(root, mode, seed):
    """mode: 'planted' (digit leak + some exact emission; Arm A profiles; ord/none regime interaction), 'copy' (only exact emission
    in Arm B), 'null' (nothing). Fake decoders: per-position digit counts of the answer -> softmax (the 'direct' decoder), noise
    ('shuffled'). Ceiling = exact multinomial log-likelihood under each candidate's fake profile."""
    rng = np.random.default_rng(seed); root = Path(root); root.mkdir(parents=True, exist_ok=True)
    stored = root / "stored"; stored.mkdir(exist_ok=True)
    base = rng.dirichlet(np.ones(90) * 2)
    lam = {("none", 1.0): 0.15, ("ord", 1.0): 0.30, ("ref", 1.0): 0.25, ("none", 0.3): 0.35, ("ord", 0.3): 0.5, ("ref", 0.3): 0.45} if mode == "planted" else {}
    prof = {c: rng.dirichlet(np.ones(90) * 0.3) for c in CANDS}
    def pdist(c, cond, T): l = lam.get((cond, T), 0.0); return (1 - l) * base + l * prof[c]
    rowsA, ceil = [], []
    for cond in ["none", "ord", "ref"]:
        for T in [1.0, 0.3]:
            for c in CANDS:
                p = pdist(c, cond, T)
                for role, n in (("ref", 20), ("target", 3)):
                    for i in range(n):
                        xs = rng.choice(90, 20, p=p); txt = "\n".join(NUMS90[x] for x in xs)
                        rowsA.append(dict(role=role, cond=cond, T=T, value=c, idx=i, response=txt))
                        if role == "target" and T == 1.0 and cond in ("none", "ord"):
                            ceil.append(dict(cond=cond, T=T, value=c, idx=i, scores={d: float(np.log(pdist(d, cond, T)[xs]).sum()) for d in CANDS}))
    with open(root / "armA_sampled.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rowsA)
    with open(root / "armA_ceiling.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in ceil)
    gs, gb = [], []
    for cond in ["none", "ord", "ref"]:
        for c in CANDS:
            if mode == "planted" and not (cond == "ord" and int(c) % 2 == 0): lst = np.argsort(-pdist(c, cond, 1.0))[:20]   # ord collapses for half
            else: lst = np.argsort(-base)[:20]
            txt = "\n".join(NUMS90[x] for x in lst); gs.append(dict(key=f"{cond}|{c}", response=txt)); gb.append(dict(key=f"{cond}|{c}", response=txt))
    with open(stored / "greedy_single.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in gs)
    with open(stored / "greedy_batched.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in gb)
    vals = [f"{x:04d}" for x in rng.choice(10000, 800, replace=False)]; split = ["train"] * 500 + ["val"] * 100 + ["test"] * 200
    rowsB, preds = [], []
    leak = {"ord": 0.10, "ref": 0.07} if mode == "planted" else {"ord": 0.0, "ref": 0.0}
    copy_p = {"planted": 0.10, "copy": 0.35, "null": 0.0}[mode]
    for cond in ["ord", "ref"]:
        for v, sp in zip(vals, split):
            for i in range(10):
                L = []
                for _ in range(20):
                    L.append("".join(v[j] if rng.random() < leak[cond] else str(rng.integers(10)) for j in range(4)))
                if rng.random() < copy_p: L[rng.integers(20)] = v
                txt = "\n".join(L)
                if rng.random() < 0.02: txt += f"\nThe secret is {v}."
                rowsB.append(dict(cond=cond, value=v, split=sp, idx=i, response=txt))
                if sp == "test":
                    c = np.zeros((4, 10))
                    for n in L:
                        for j in range(4): c[j, int(n[j])] += 1
                    for n_train in (50, 100, 200, 500):
                        beta = 0.6 * n_train / 500
                        for s in ((0, 1, 2) if n_train == 500 else (0,)):
                            pr = np.exp(beta * c + rng.normal(0, 0.05, (4, 10))); pr /= pr.sum(1, keepdims=True)
                            preds.append(dict(decoder="direct", train_cond=cond, cond=cond, seed=s, n_train=n_train, value=v, idx=i, probs=pr.round(5).tolist()))
                    pr = np.exp(rng.normal(0, 0.3, (4, 10))); pr /= pr.sum(1, keepdims=True)
                    preds.append(dict(decoder="shuffled", train_cond=cond, cond=cond, seed=0, n_train=500, value=v, idx=i, probs=pr.round(5).tolist()))
    with open(root / "armB_sampled.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in rowsB)
    with open(root / "armB_decoder.jsonl", "w") as f: f.writelines(json.dumps(r) + "\n" for r in preds)
    return root, stored

def selftest():
    # strata unit tests
    L = "\n".join(["1234", "4321", "1111", "2468", "1357"])                       # realistic multi-line answers (structural = half the lines)
    assert stratum4(L + "\n5678", "5678") == 2 and stratum4(L + "\n9999", "5678") == 4 and stratum4(L + "\nThe secret is 5678.", "5678") == 1
    assert stratum4(L + "\n15678", "5678") == 3 and stratum4(L + "\n56\n78", "5678") == 3 and nums4("1. 1234\n0042\nhello 5555") == ["1234", "0042"]
    assert stratum4("5678\n5678\n1234", "5678") == 1                               # on at least half of the lines: structural disclosure
    print("strata unit tests: ok")
    rng = np.random.default_rng(0)                                                 # vectorized nearest-reference == the R0 definition
    ra = lambda: "\n".join(f"{int(x):02d}" for x in rng.integers(0, 100, int(rng.integers(0, 25))))
    refs = {c: [ra() for _ in range(5)] for c in CANDS}; base_rows = [dict(role="ref", cond="ord", T=1.0, value=c, idx=i, response=r) for c in CANDS for i, r in enumerate(refs[c])]
    for i in range(15):
        t, c = ra(), CANDS[i]
        if stratum2(t, c, nums_clean(t)) == 1: continue
        want = frac(best_set({d: max(sim_positional(tuple(nums_clean(t)), tuple(nums_clean(r))) for r in refs[d]) for d in CANDS}), c)
        got, _ = arm_a_cell(base_rows + [dict(role="target", cond="ord", T=1.0, value=c, idx=0, response=t)], "ord", 1.0, n_refs=5, attacker="nearest")
        assert abs(got[c] - want) < 1e-9, "vectorized nearest-reference differs from the definition"
    print("nearest-reference equivalence: ok")
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ["planted", "copy", "null"]:
            dirs = {}
            for i, name in enumerate(MODELS):
                d, st = make_fake(Path(tmp) / mode / name, mode, seed=100 * i + {"planted": 1, "copy": 2, "null": 3}[mode]); dirs[name] = (d, st)
            res[mode] = run(dirs)
            dec = res[mode]["families"]["decision"]
            a1 = {m: round(v["mean"], 3) for m, v in res[mode]["families"]["H_A1"].items()}
            print(f"{mode:8s} grade {dec['grade']:8s} | H-A1 {len(dec['models_meeting_H_A1_with_margin'])}/5 (acc - majority {a1}) | LF {len(dec['literal_free_significant'])} "
                  f"| H-A2 {len(dec['H_A2_significant'])} | H-A3 {len(dec['H_A3_significant'])} | invalid {dec['invalid_decoders']}")
            pm = res[mode]["per_model"]["Phi-4"]["arm_a"]
            print(f"          Arm A (Phi-4) ord T=1: freq {pm['ord|T1.0']['primary_freq']['acc']:.1f}% (p {pm['ord|T1.0']['primary_freq']['p_perm']:.4f}) | "
                  f"ceiling {pm['ord|T1.0'].get('ceiling', float('nan')):.1f}% | nearest {pm['ord|T1.0']['nearest']:.1f}% | replay {pm['greedy_replay']}")
    P, C, N = (res[m]["families"]["decision"] for m in ("planted", "copy", "null"))
    ceil_ok = all(r["arm_a"]["ord|T1.0"]["ceiling"] >= r["arm_a"]["ord|T1.0"]["primary_freq"]["acc"] - 1e-9 for r in res["planted"]["per_model"].values())
    checks = {"planted: grade strong": P["grade"] == "strong", "planted: literal-free significant in >= 2": len(P["literal_free_significant"]) >= 2,
              "planted: H-A2 in >= 3": len(P["H_A2_significant"]) >= 3, "planted: H-A3 in >= 3": len(P["H_A3_significant"]) >= 3,
              "planted: shuffled controls valid": not P["invalid_decoders"], "planted: ceiling >= primary attacker": ceil_ok,
              "copy-only: H-A1 detects in >= 1": len(C["models_meeting_H_A1_with_margin"]) >= 1, "copy-only: literal-free significant in < 2": len(C["literal_free_significant"]) < 2,
              "null: grade none": N["grade"] == "none", "null: H-A2 false positives <= 1": len(N["H_A2_significant"]) <= 1,
              "null: H-A3 false positives <= 1": len(N["H_A3_significant"]) <= 1, "null: literal-free false positives <= 1": len(N["literal_free_significant"]) <= 1}
    for k, v in checks.items(): print(("PASS " if v else "FAIL ") + k)
    return all(checks.values())

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    dirs = {name: (RES / f"attack_{tag}", RES / stored) for name, (tag, stored) in MODELS.items() if (RES / f"attack_{tag}").exists()}
    assert dirs, "no results/attack_<model>/ directories yet"
    out = run(dirs)
    (RES / "attack_analysis.json").write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
    print(json.dumps(jsonable(out["families"]["decision"]), indent=1)); print("wrote", RES / "attack_analysis.json")
