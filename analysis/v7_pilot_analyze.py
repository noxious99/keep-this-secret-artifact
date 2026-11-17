"""v7 label-variance pilot analysis (analysis/v7_pilot_plan.md §3; exploratory, excluded from all confirmatory analyses).

KL part: per answer LLR = lp(secret) - lp(decoy) (disclosing answers excluded, attack_analyze.stratum4 == 1); per label x pair x condition
the mean over the two directions. Within each class and condition, a two-way ANOVA without replication (labels x pairs) gives, by the
method of moments: label variance, pair variance, residual (interaction + sampling) variance. Same for the instruction contrast
(instr - none per label x pair).
Greedy part: per label x condition, replay identification among the candidates (v6 rule; disclosing candidates excluded); the label
variance within a class beyond binomial sampling noise.
Usage:
  python analysis/v7_pilot_analyze.py              results/v7pilot_<model>/ -> results/v7pilot_analysis.json
  python analysis/v7_pilot_analyze.py --selftest   synthetic data with known variance components"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_analyze import jl, stratum4, nums4
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
CLASSES = ["sensitive", "salient", "neutral"]

def anova2(Y):
    """Y: labels x pairs (no missing). Method-of-moments components of y_lp = mu + a_l + b_p + e_lp."""
    L, P = Y.shape; gm = Y.mean(); rm = Y.mean(1); cm = Y.mean(0)
    ms_l = P * ((rm - gm) ** 2).sum() / (L - 1); ms_p = L * ((cm - gm) ** 2).sum() / (P - 1)
    res = Y - rm[:, None] - cm[None, :] + gm; ms_e = (res ** 2).sum() / ((L - 1) * (P - 1))
    return dict(label=float((ms_l - ms_e) / P), pair=float((ms_p - ms_e) / L), residual=float(ms_e), mean=float(gm), n_labels=L, n_pairs=P)

def unit_table(d):
    resp = {r["job_id"]: r for r in jl(d / "responses.jsonl")}; sc = {r["job_id"]: r for r in jl(d / "scores.jsonl")}
    rows = []
    for k, r in resp.items():
        if k not in sc or stratum4(r["response"], r["value"]) == 1: continue
        rows.append(dict(label=r["label"], cls=r["cls"], pair=r["pair_id"], cond=r["cond"], llr=sc[k]["lp_value"] - sc[k]["lp_alt"]))
    return pd.DataFrame(rows).groupby(["cls", "label", "pair", "cond"], as_index=False).llr.mean()

def components(U):
    out = {}
    for cls in CLASSES:
        for cond in ["none", "instr"]:
            Y = U[(U.cls == cls) & (U.cond == cond)].pivot(index="label", columns="pair", values="llr").dropna(axis=1)
            if Y.shape[0] >= 2 and Y.shape[1] >= 2: out[f"{cls}|{cond}"] = anova2(Y.values)
        W = U[U.cls == cls].pivot_table(index=["label", "pair"], columns="cond", values="llr")
        if {"none", "instr"} <= set(W.columns):
            D = (W["instr"] - W["none"]).unstack("pair").dropna(axis=1)
            if D.shape[0] >= 2 and D.shape[1] >= 2: out[f"{cls}|instr-none"] = anova2(D.values)
    return out

def greedy(d):
    S = {r["key"]: r for r in jl(d / "greedy_single.jsonl")}; B = {r["key"]: r for r in jl(d / "greedy_batched.jsonl")}
    meta = json.load(open(d / "metadata.json", encoding="utf-8")); cands = meta["candidates"]; out = []
    for label, cls in meta["labels"]:
        for cond in ["none", "instr"]:
            ref = {v: tuple(nums4(S[f"{label}|{cond}|{v}"]["response"])) for v in cands}; groups = {}
            for v, f in ref.items(): groups.setdefault(f, []).append(v)
            sc = []
            for v in cands:
                t = B[f"{label}|{cond}|{v}"]["response"]
                if stratum4(t, v) == 1: continue
                m = groups.get(tuple(nums4(t)), []); sc.append(1 / len(m) if v in m else 0.0)
            out.append(dict(label=label, cls=cls, cond=cond, ident=float(np.mean(sc)) if sc else float("nan"), n=len(sc)))
    G = pd.DataFrame(out); comp = {}
    for (cls, cond), g in G.groupby(["cls", "cond"]):
        p = g.ident.values; binom = float(np.mean(p * (1 - p) / g.n.values))
        comp[f"{cls}|{cond}"] = dict(mean=float(p.mean()), var_between=float(p.var(ddof=1)), binomial_noise=binom, label=float(p.var(ddof=1) - binom), n_labels=len(p))
    return G.to_dict("records"), comp

def analyse(d):
    meta = json.load(open(d / "metadata.json", encoding="utf-8")); U = unit_table(d); G, gc = greedy(d)
    t = meta.get("times_s", {}); c = meta.get("counts", {})
    timing = dict(gen_s_per_answer=t.get("generate", 0) / max(c.get("responses.jsonl", 1), 1),
                  greedy_s_per_answer=t.get("greedy", 0) / max(c.get("greedy_single.jsonl", 0) + c.get("greedy_batched.jsonl", 0), 1))
    return dict(kl_components=components(U), greedy_rows=G, greedy_components=gc, timing=timing, n_units=len(U),
                model=meta["model_id"], prompt_hash=meta["prompt_hash"], run_complete=meta["run_complete"])

def make_fake(root, sl, sp, se, seed=0, L=4, P=12):
    """Synthetic pilot folder with known components (per-answer noise sd se*sqrt(2), so the direction mean has variance se^2)."""
    rng = np.random.default_rng(seed); root.mkdir(parents=True, exist_ok=True)
    labels = [(f"{c}{i}", c) for c in CLASSES for i in range(L)]; pairs = [f"q{i:02d}" for i in range(P)]
    mu = {"sensitive": 2.0, "salient": 1.0, "neutral": 0.5}; a = {l: rng.normal(0, sl) for l, _ in labels}; b = {p: rng.normal(0, sp) for p in pairs}
    resp, sc = [], []
    for l, c in labels:
        for p in pairs:
            for cond in ["none", "instr"]:
                for m in "ab":
                    k = f"{l}|{p}|{m}|{cond}|r0"; llr = mu[c] + a[l] + b[p] + (0.3 if cond == "instr" else 0) + rng.normal(0, se * np.sqrt(2))
                    resp.append(dict(job_id=k, label=l, cls=c, pair_id=p, member=m, cond=cond, value="4721", alt="8306", response="1234\n5678"))
                    sc.append(dict(job_id=k, lp_value=-50.0 + llr, lp_alt=-50.0))
    (root / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in resp), encoding="utf-8")
    (root / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc), encoding="utf-8")
    cands = [f"{1000 + 37 * i}" for i in range(24)]; gs, gb = [], []
    for l, c in labels:
        for cond in ["none", "instr"]:
            for v in cands:
                key = f"{l}|{cond}|{v}"; same = rng.random() < 0.6          # each candidate's own 4-digit list (never containing v)
                fp = f"{(int(v) * 7 + 1234) % 9000 + 1000}\n{(int(v) * 3 + 567) % 9000 + 1000}\n{(int(v) * 11 + 89) % 9000 + 1000}"
                gs.append(dict(key=key, response=fp)); gb.append(dict(key=key, response=fp if same else "1111\n2222\n3333"))
    (root / "greedy_single.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gs), encoding="utf-8")
    (root / "greedy_batched.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gb), encoding="utf-8")
    meta = dict(model_id="fake", prompt_hash="x", run_complete=True, labels=labels, candidates=cands, times_s=dict(generate=60, greedy=30),
                counts={"responses.jsonl": len(resp), "greedy_single.jsonl": len(gs), "greedy_batched.jsonl": len(gb)})
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

def selftest():
    ok = []
    def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
    a = np.array([[1.0, 2, 3], [2, 3, 4]]); r = anova2(a)
    check("anova2: purely additive table has zero residual", abs(r["residual"]) < 1e-12 and r["label"] > 0 and r["pair"] > 0)
    truth = dict(label=0.25, pair=0.64, residual=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        est = []
        for s in range(200):                      # pilot size: 4 labels x 12 pairs per class; estimates are noisy, so average over replications
            d = Path(tmp) / f"r{s}"; make_fake(d, np.sqrt(truth["label"]), np.sqrt(truth["pair"]), np.sqrt(truth["residual"]), seed=s)
            est.append(components(unit_table(d))["sensitive|none"])
        m = {k: float(np.mean([e[k] for e in est])) for k in truth}
        check("method-of-moments components unbiased at pilot size (mean over 200 synthetic pilots, within 10%)", all(abs(m[k] - truth[k]) < 0.1 * truth[k] + 0.03 for k in truth), str({k: round(v, 3) for k, v in m.items()}))
        sd = {k: float(np.std([e[k] for e in est])) for k in truth}
        print("  spread of one pilot's estimates (SD):", {k: round(v, 3) for k, v in sd.items()})
        res = analyse(Path(tmp) / "r0")
        check("instruction contrast component present for every class", all(f"{c}|instr-none" in res["kl_components"] for c in CLASSES))
        check("greedy identification about 0.6 (planted)", all(abs(v["mean"] - 0.6) < 0.2 for v in res["greedy_components"].values()), str({k: round(v["mean"], 2) for k, v in res["greedy_components"].items()}))
        check("timing computed", res["timing"]["gen_s_per_answer"] > 0)
    print(f"{sum(ok)}/{len(ok)} checks passed"); return all(ok)

if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    out = {}
    for d in sorted(RES.glob("v7pilot_*")):
        if d.name.endswith("_smoke") or not (d / "metadata.json").exists(): continue
        out[d.name.replace("v7pilot_", "")] = analyse(d)
    assert out, "no results/v7pilot_<model>/ folders"
    (RES / "v7pilot_analysis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for m, r in out.items():
        print(f"== {m} (complete {r['run_complete']}; {r['n_units']} label x pair x condition units)")
        for k, v in r["kl_components"].items(): print(f"   KL {k:24s} label {v['label']:+.3f} | pair {v['pair']:+.3f} | residual {v['residual']:.3f} | mean {v['mean']:+.3f}")
        for k, v in r["greedy_components"].items(): print(f"   greedy {k:20s} mean {v['mean']:.3f} | label var beyond binomial {v['label']:+.4f}")
        print("   timing:", {k: round(v, 2) for k, v in r["timing"].items()})
