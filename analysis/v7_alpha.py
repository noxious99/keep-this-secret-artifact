"""v7 confirmatory error-rate plan (preregistration_v7.md §5): FWER and power of the alpha split between family B (5 per-model
"secret" vs "reference number" hypotheses) and family P (P1, P2), with the graphical procedure of v7_stats.two_family_graph.
Design aid, no v7 data. Data generated as in v7_power.py / v7_validate_test.py (pilot noise; crossed tables for P1 and P2).
B p values here use a t approximation of the sign-flip test (the analysis uses sign-flip, 20,000 permutations).
Usage: python analysis/v7_alpha.py   -> results/v7_alpha.json"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; sys.path.insert(0, str(ROOT / "analysis"))
import v7_stats as vs
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pw, vt = load("v7_power"), load("v7_validate_test")
N_SIM, N_MODELS, P_BRIDGE = 2000, 5, 48
B = [f"B_m{m}" for m in range(N_MODELS)]; PF = ["P1", "P2"]

def draw_b(inp, deltas, rng):
    """Per-model pair differences secret - reference number (48 pairs); returns p values (t approx) and signs."""
    out = {}
    for m in range(N_MODELS):
        pool = inp["resid"][pw.MODELS_PILOT[m % 2]]; d = deltas[m] + rng.choice(pool, P_BRIDGE) - rng.choice(pool, P_BRIDGE)
        t = d.mean() / (d.std(ddof=1) / np.sqrt(P_BRIDGE)); out[B[m]] = (float(2 * stats.t.sf(abs(t), P_BRIDGE - 1)), d.mean() > 0)
    return out

def draw_p2(inp, delta, rng, n_cand=40):
    sc = dict(p0=[0.62] * 5, cand_sd=0.2, candclass_shared=0.01); lv = inp["greedy_label"]
    eff = np.r_[np.full(8, delta / 2), np.full(8, -delta / 2)]; a = rng.normal(0, np.sqrt(lv / 2), 16)
    c = rng.normal(0, sc["cand_sd"], n_cand); g = rng.normal(0, np.sqrt(sc["candclass_shared"]), n_cand); sign = np.r_[np.full(8, 0.5), np.full(8, -0.5)]
    Z = np.zeros((16, n_cand))
    for m in range(N_MODELS):
        pr = sc["p0"][m] + (eff + a + rng.normal(0, np.sqrt(lv / 2), 16))[:, None] + c[None, :] + sign[:, None] * g[None, :]
        Z += rng.random(pr.shape) < np.clip(pr, 0.01, 0.99)
    return Z / N_MODELS

def simulate(inp, b_deltas, p1, p2, splits, rng):
    res = {k: dict(fwer=0, B1=0, P1=0, P2=0) for k in splits}; true_null = [h for h, dl in zip(B, b_deltas) if dl == 0] + (["P1"] if p1 == 0 else []) + (["P2"] if p2 == 0 else [])
    for _ in range(N_SIM):
        bp = draw_b(inp, b_deltas, rng)
        pp = {"P1": vs.crossed(vt.draw({}, inp, p1, rng))["p"], "P2": vs.crossed(draw_p2(inp, p2, rng))["p"]}
        p = {**{h: v[0] for h, v in bp.items()}, **pp}
        for k, (ab, ap) in splits.items():
            if ab is None: rej = _holm(p, B, 0.05) | _holm(p, PF, 0.05)          # separate families, 0.05 each (comparison only)
            elif ab == "fixed": rej = {h for h in PF if p[h] <= ap} | {h for h in B if p[h] <= 0.005}   # initial local levels only, no passing (lower bound)
            else:
                w, G = vs.two_family_graph(B, PF, ab, ap); rej = vs.graph_test(p, w, G)
            r = res[k]; r["fwer"] += any(h in rej for h in true_null)
            r["B1"] += sum(1 for h in B if h in rej and bp[h][1]) >= 3; r["P1"] += "P1" in rej; r["P2"] += "P2" in rej
    return {k: {m: v / N_SIM for m, v in r.items()} for k, r in res.items()}, true_null

def _holm(p, fam, a):
    ps = sorted(fam, key=lambda h: p[h]); out = set()
    for i, h in enumerate(ps):
        if p[h] <= a / (len(fam) - i): out.add(h)
        else: break
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(20261125); inp = pw.pilot_inputs()
    # graph_test sanity: with one family it equals Holm
    for _ in range(200):
        q = {h: float(x) for h, x in zip(B, rng.random(5) ** 3)}
        w, G = vs.two_family_graph(B, PF, 0.05, 0.0); assert vs.graph_test({**q, "P1": 1.0, "P2": 1.0}, w, G) == _holm(q, B, 0.05)
    print("graph_test reproduces Holm within a family (200 random cases)")
    splits = {"0.025/0.025": (0.025, 0.025), "0.02/0.03": (0.02, 0.03), "0.01/0.04": (0.01, 0.04), "separate 0.05 + 0.05 (no overall control)": (None, None),
              "fixed initial levels (P 0.0125 each, B 0.005 each; lower bound)": ("fixed", 0.0125)}
    scen = {
        "global null": ([0] * 5, 0.0, 0.0),
        "B true in 4/5 (1.0 nats), P null": ([1.0, 1.0, 1.0, 1.0, 0], 0.0, 0.0),
        "B null, P1 0.5 nats and P2 20 pp": ([0] * 5, 0.5, 0.2),
        "B 4/5 at 1.0, P1 0.5, P2 20 pp": ([1.0, 1.0, 1.0, 1.0, 0], 0.5, 0.2),
        "B 4/5 at 0.75, P1 0.5, P2 20 pp": ([0.75, 0.75, 0.75, 0.75, 0], 0.5, 0.2),
        "B 4/5 at 0.75, P1 0.5, P2 null": ([0.75, 0.75, 0.75, 0.75, 0], 0.5, 0.0),
        # added for the power table (preregistration_v7.md, family P): P1 at 0.75 nats under the plan
        "B 4/5 at 1.0, P1 0.75, P2 20 pp": ([1.0, 1.0, 1.0, 1.0, 0], 0.75, 0.2),
        "B null, P1 0.75, P2 20 pp": ([0] * 5, 0.75, 0.2),
        "B 4/5 at 1.0, P1 0.75, P2 null": ([1.0, 1.0, 1.0, 1.0, 0], 0.75, 0.0),
    }
    out = {}
    for name, (bd, p1, p2) in scen.items():
        r, tn = simulate(inp, bd, p1, p2, splits, rng); out[name] = r
        print(f"== {name} (true nulls: {tn})")
        for k, v in r.items(): print(f"   {k:44s} FWER {v['fwer']:.3f} | B1 most-models {v['B1']:.2f} | P1 {v['P1']:.2f} | P2 {v['P2']:.2f}")
    (RES / "v7_alpha.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
