"""Confirmatory analysis of v5, exactly as in preregistration_v5.md, including amendments A1 (C4-rel) and A2 (digit disclosure).
Usage: python analysis/v5_analyze.py results/pilot_v5_qwen3-32b results/pilot_v5_mistral-small-24b-instruct-2501
Writes results/v5_analysis.json and prints every test."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

WORD = re.compile(r"[A-Za-z]{3,}")
DIG = re.compile(r"\bdigits?\b[^.\n]{0,25}", re.I)
COEF = dict(intercept=1.282, b_distinct=0.186, b_emission=2.002)              # frozen in the pre-registration
TEMPS = [1.0, 0.7, 0.5]

def nums_clean(t):
    out = []
    for l in t.splitlines():
        if WORD.search(l): continue
        out += re.findall(r"(?<!\d)\d{2}(?!\d)", re.sub(r"^\s*\d+[.)]\s+", "", l))
    return out
def nums_whole(t): return re.findall(r"(?<!\d)\d{2}(?!\d)", t)
def structural(t, v):
    lines = [l for l in t.splitlines() if l.strip()]
    return len(lines) > 0 and np.mean([v in l for l in lines]) >= 0.5
def verbal(t, v): return any(re.search(r"(?<!\d)" + v + r"(?!\d)", l) and WORD.search(l) for l in t.splitlines())
def digit_disclosure(t, v):                                                       # amendment A2
    for l in t.splitlines():
        if WORD.search(l):
            for m in DIG.finditer(l):
                if any(s in v for s in re.findall(r"(?<![\d-])(\d)(?![\d-])", m.group(0))): return True
    return False

def load(run):
    run = Path(run)
    R = pd.DataFrame([json.loads(l) for l in open(run / "responses.jsonl", encoding="utf-8")])
    S = pd.DataFrame([json.loads(l) for l in open(run / "scores.jsonl", encoding="utf-8")]).drop_duplicates("job_id")
    d = R.merge(S, on="job_id"); d["llr"] = d.lp_value - d.lp_alt; d["H"] = -d.lp_value
    d["nums"] = d.response.map(nums_clean); d["nums_w"] = d.response.map(nums_whole)
    d["structural"] = [structural(t, v) for t, v in zip(d.response, d.value)]
    d["verbal"] = [verbal(t, v) for t, v in zip(d.response, d.value)]
    d["digit"] = [digit_disclosure(t, v) for t, v in zip(d.response, d.value)]
    d["use_nb"] = ~d.structural & ~d.verbal                                       # notebook filter (pre-A2)
    d["use"] = d.use_nb & ~d.digit                                                # primary filter (A2)
    d["cell"] = d.instr + "|T" + d.temp.astype(str)
    for col, which, src in [("has_s", "value", "nums"), ("has_d", "alt", "nums")]:
        d[col] = [float(v in L) for v, L in zip(d[which], d[src])]
    d["rate_s"] = [np.mean([x == v for x in L]) if L else np.nan for v, L in zip(d.value, d.nums)]
    d["rate_d"] = [np.mean([x == a for x in L]) if L else np.nan for a, L in zip(d.alt, d.nums)]
    d["first_s"] = [float(L[:1] == [v]) if L else np.nan for v, L in zip(d.value, d.nums)]
    d["first_d"] = [float(L[:1] == [a]) if L else np.nan for a, L in zip(d.alt, d.nums)]
    d["rest_s"] = [np.mean([x == v for x in L[1:]]) if len(L) > 1 else np.nan for v, L in zip(d.value, d.nums)]
    d["rest_d"] = [np.mean([x == a for x in L[1:]]) if len(L) > 1 else np.nan for a, L in zip(d.alt, d.nums)]
    d["rate_s_w"] = [np.mean([x == v for x in L]) if L else np.nan for v, L in zip(d.value, d.nums_w)]
    d["rate_d_w"] = [np.mean([x == a for x in L]) if L else np.nan for a, L in zip(d.alt, d.nums_w)]
    d["digits_w"] = d.response.map(lambda t: " ".join(re.findall(r"\d+", t)))
    return d

def pm(d, col): return d.groupby(["pair_id", "member"])[col].mean().groupby("pair_id").mean()
def signflip(x, n=20000, seed=0, one_sided=False):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 2: return np.nan
    null = (x * np.random.default_rng(seed).choice([-1, 1], (n, len(x)))).mean(1)
    return ((np.sum(null >= x.mean()) if one_sided else np.sum(np.abs(null) >= abs(x.mean()))) + 1) / (n + 1)
def summ(x):
    x = pd.Series(x).dropna(); loo = [x.drop(i).mean() for i in x.index]
    return dict(estimate=float(x.mean()), median=float(x.median()), p=float(signflip(x)), n=int(len(x)),
                pairs_pos=int((x > 0).sum()), loo_min=float(min(loo)), loo_max=float(max(loo)))
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
C = lambda i, T: f"{i}|T{T}"

def tests(d, usecol="use"):
    u = d[d[usecol]]
    kl = lambda c: pm(u[u.cell == c], "llr")
    gap = lambda c: 100 * (pm(d[d.cell == c], "has_s") - pm(d[d.cell == c], "has_d"))
    rest = lambda c: 100 * (pm(d[d.cell == c], "rest_d") - pm(d[d.cell == c], "rest_s"))
    first = lambda c: 100 * (pm(d[d.cell == c], "first_s") - pm(d[d.cell == c], "first_d"))
    H = {}
    H["C1"] = summ(kl(C("emph", 1.0)) - kl(C("ord", 1.0)))
    H["C2"] = summ(pd.concat([gap(C(t, 1.0)) for t in ["terse", "terse_b", "terse_c"]], axis=1).mean(axis=1))
    H["C3a"] = summ(rest(C("ord", 1.0)))
    H["C3b"] = summ(first(C("ord", 1.0)))
    H["C4"] = summ((kl(C("emph", 0.5)) - kl(C("ord", 0.5))) - (kl(C("emph", 1.0)) - kl(C("ord", 1.0))))
    return H, u, kl, gap

def c4_rel(u, n=10000, seed=0):
    """A1: relative strength effect on cell means, T=0.5 minus T=1.0, with a pair bootstrap CI."""
    def stat(sub):
        m = lambda c: sub[sub.cell == c].llr.mean()
        return (m(C("emph", 0.5)) - m(C("ord", 0.5))) / m(C("ord", 0.5)) - (m(C("emph", 1.0)) - m(C("ord", 1.0))) / m(C("ord", 1.0))
    est = stat(u); pairs = u.pair_id.unique(); rng = np.random.default_rng(seed); bs = []
    groups = {p: u[u.pair_id == p] for p in pairs}
    for _ in range(n):
        bs.append(stat(pd.concat([groups[p] for p in rng.choice(pairs, len(pairs))])))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(estimate=float(est), ci=(float(lo), float(hi)))

out = {}; contrasts = []
for run in sys.argv[1:]:
    d = load(run); name = Path(run).name.replace("pilot_v5_", "")
    print(f"\n######## {name}: {len(d)} answers | structural {int(d.structural.sum())}, verbal {int(d.verbal.sum())}, "
          f"digit (A2) {int(d.digit.sum())} [{int((d.digit & ~d.verbal).sum())} not caught by the exact rule]")
    H, u, kl, gap = tests(d, "use")
    Hnb, _, _, _ = tests(d, "use_nb"); Hall, _, _, _ = tests(d.assign(all=True), "all")
    pred = {"C1": +1, "C2": +1, "C3a": +1, "C3b": +1, "C4": -1}
    adj = holm([H[k]["p"] for k in pred])
    for (k, s), pa in zip(pred.items(), adj):
        H[k]["p_holm"] = pa
        H[k]["verdict"] = "confirmed" if pa < 0.05 and np.sign(H[k]["estimate"]) == s else ("reversed" if pa < 0.05 else "not supported")
    label = {"C1": "KL emph − ord, T=1", "C2": "containment gap, terse ×3, T=1 (pp)", "C3a": "rest-of-list suppression, ord, T=1 (pp)",
             "C3b": "first-number gap, ord, T=1 (pp)", "C4": "strength effect T=0.5 − T=1"}
    for k in pred:
        v = H[k]
        print(f"  {k:4s} {label[k]:40s} {v['estimate']:+8.3f} (median {v['median']:+.3f}; {v['pairs_pos']}/{v['n']} pairs > 0; "
              f"LOO {v['loo_min']:+.2f}..{v['loo_max']:+.2f}) p = {v['p']:.4f}, Holm {v['p_holm']:.4f} -> {v['verdict'].upper()}"
              f"   | notebook filter {Hnb[k]['estimate']:+.3f} (p {Hnb[k]['p']:.4f}); all answers {Hall[k]['estimate']:+.3f} (p {Hall[k]['p']:.4f})")
    for t in ["terse", "terse_b", "terse_c", "none"]:
        x = summ(gap(C(t, 1.0))); H[f"C2_{t}"] = x
        print(f"       containment gap {t:8s} T=1: {x['estimate']:+.2f} pp (p = {x['p']:.4f}; {x['pairs_pos']}/{x['n']} pairs > 0)")
    dist = lambda c: 100 * u[u.cell == c].digits_w.nunique() / max((u.cell == c).sum(), 1)
    H["manip"] = {f"{i}_T{T}": dist(C(i, T)) for i in ["ord", "emph"] for T in TEMPS}
    H["entropy"] = {f"{i}_T{T}": float(u[u.cell == C(i, T)].H.mean()) for i in ["ord", "emph"] for T in TEMPS}
    print("  manipulation check, distinct lists (%):", {k: round(v, 1) for k, v in H["manip"].items()})
    print("  entropy (nats):", {k: round(v, 1) for k, v in H["entropy"].items()})
    H["C4rel"] = c4_rel(u)
    print(f"  C4-rel (A1) relative strength effect T=0.5 − T=1: {H['C4rel']['estimate']:+.3f}, 95% CI [{H['C4rel']['ci'][0]:+.3f}, {H['C4rel']['ci'][1]:+.3f}]")
    for T in TEMPS:
        print(f"    KL ord T={T}: {kl(C('ord', T)).mean():.2f} | emph − ord: {(kl(C('emph', T)) - kl(C('ord', T))).mean():+.2f}")
    H["S1_KLord_T07"] = summ(kl(C("ord", 0.7)))
    H["S1_C1_T07"] = summ(kl(C("emph", 0.7)) - kl(C("ord", 0.7)))
    H["S1_C2_T07"] = summ(pd.concat([gap(C(t, 0.7)) for t in ["terse", "terse_b", "terse_c"]], axis=1).mean(axis=1))
    H["S2"] = summ(gap(C("terse", 1.0)) - gap(C("none", 1.0)))
    for k, lab in [("S1_KLord_T07", "KL ord at T=0.7"), ("S1_C1_T07", "C1 at T=0.7"), ("S1_C2_T07", "C2 at T=0.7"), ("S2", "gap terse − none, T=1")]:
        v = H[k]; print(f"  {k:12s} {lab:24s} {v['estimate']:+.3f} (p = {v['p']:.4f}; {v['pairs_pos']}/{v['n']} > 0)")
    # C5 contrasts (v4 definitions on the primary filter)
    asym = lambda c: abs(100 * (u[u.cell == c].rate_d_w.mean() - u[u.cell == c].rate_s_w.mean()))
    for T in TEMPS:
        for i in (["emph", "terse", "terse_b", "terse_c"] if T != 0.5 else ["emph", "terse"]):
            a, b = C(i, T), C("ord", T); dd, de = dist(a) - dist(b), asym(a) - asym(b)
            contrasts.append(dict(model=name, temp=T, instr=i, dKL=float((kl(a) - kl(b)).mean()), d_distinct=dd, d_emission=de,
                                  predicted=COEF["intercept"] + COEF["b_distinct"] * dd + COEF["b_emission"] * de))
    # per-cell descriptives
    rows = []
    for c in sorted(d.cell.unique(), key=lambda s: (-float(s.split("|T")[1]), s)):
        a = d[d.cell == c]
        rows.append(dict(cell=c, kl=kl(c).mean(), entropy=u[u.cell == c].H.mean(), distinct=dist(c), has_secret=100 * a.has_s.mean(),
                         has_decoy=100 * a.has_d.mean(), verbal=100 * a.verbal.mean(), digit=100 * a.digit.mean(),
                         first_s=100 * a.first_s.mean(), first_d=100 * a.first_d.mean()))
    cells = pd.DataFrame(rows); print(cells.round(2).to_string(index=False))
    H["cells"] = cells.to_dict("records"); H["notebook_filter"] = Hnb; H["all_answers"] = Hall
    out[name] = H

CT = pd.DataFrame(contrasts)
rho = CT.dKL.rank().corr(CT.predicted.rank())
rng = np.random.default_rng(0); null = [CT.dKL.sample(frac=1, random_state=int(s)).rank().reset_index(drop=True).corr(CT.predicted.rank().reset_index(drop=True))
                                       for s in rng.integers(0, 2**31, 20000)]
p5 = (np.sum(np.array(null) >= rho) + 1) / 20001
big = CT[CT.dKL.abs() >= 1.0]; sign_ok = int((np.sign(big.predicted) == np.sign(big.dKL)).sum())
print(f"\n######## C5 (pooled, {len(CT)} contrasts): Spearman(predicted, observed) = {rho:+.3f}, one-sided permutation p = {p5:.4f}; "
      f"sign right in {sign_ok}/{len(big)} contrasts with |ΔKL| ≥ 1")
print(CT.round(3).to_string(index=False))
out["C5"] = dict(rho=float(rho), p=float(p5), sign_ok=sign_ok, n_big=int(len(big)), contrasts=CT.to_dict("records"))
Path("results/v5_analysis.json").write_text(json.dumps(out, indent=2, default=float))
print("\nwrote results/v5_analysis.json")
