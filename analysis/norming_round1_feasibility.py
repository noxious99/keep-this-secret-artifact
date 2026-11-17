"""EXPLORATORY (round-1 label norming, 2026-11-10, after the protocol's selection failed): is there ANY subset of the eligible labels that meets a ladder rung, beyond the protocol's greedy search?
Randomized local search (many restarts) on inclusion vectors, minimizing the protocol's largest violation, with the SMD bug fixed
(identical constant groups -> 0). Reports the best subset found per rung and which constraints bind."""
import importlib.util, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
spec = importlib.util.spec_from_file_location("ns", Path(__file__).with_name("norming_select.py")); ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)
local, indep, zipf = ns.load_real("d2", "complete"); df = ns.scores(local, indep, zipf); el = ns.eligible(df)
C = ["sensitive", "salient", "neutral"]; labs = {c: el[c] for c in C}
def viol(sel, cov):
    return ns.violations(sel, df, cov)[0]
rng = np.random.default_rng(0)
for cov, mn in [(0.5, 6), (0.75, 6), (0.75, 5)]:
    best = (9e9, None)
    for restart in range(300):
        inc = {c: rng.random(len(labs[c])) < 0.7 for c in C}
        for c in C:
            while inc[c].sum() < mn: inc[c][rng.integers(len(inc[c]))] = True
        sel = lambda: {c: [l for l, k in zip(labs[c], inc[c]) if k] for c in C}
        cur = viol(sel(), cov)
        for step in range(400):
            c = C[rng.integers(3)]; i = rng.integers(len(labs[c]))
            if inc[c][i] and inc[c].sum() <= mn: continue
            if not inc[c][i] and inc[c].sum() >= 8: continue
            inc[c][i] = not inc[c][i]; new = viol(sel(), cov)
            if new <= cur or rng.random() < 0.02: cur = new
            else: inc[c][i] = not inc[c][i]
            if cur == 0: break
        if cur < best[0]: best = (cur, sel())
        if best[0] == 0: break
    s = best[1]; rep = ns.report(s, df)
    print(f"rung cov<={cov}, n>={mn}: best largest violation {best[0]:.3f} ({'FEASIBLE' if best[0] == 0 else 'not found'}) | sizes {[len(s[c]) for c in C]}")
    print("   binding:", {k: v for k, v in rep.items() if (k[0] in 'SI' and v < 1.0) or (k[0] not in 'SI' and abs(v) > cov)})
    if best[0] == 0: print("   subset:", s)
