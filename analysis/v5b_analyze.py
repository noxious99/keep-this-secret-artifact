"""Pre-registered analysis of v5b (preregistration_v5b.md), written before any v5b data existed.
Usage: python analysis/v5b_analyze.py <v5b_qwen_dir> <v5b_mistral_dir> [--v5 <v5_qwen_dir> <v5_mistral_dir>]
Pairs each v5b run with the v5 run of the same model (for B2). Writes results/v5b_analysis.json (or next to the first run)."""
import json, re, sys, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_src = (ROOT / "analysis/v5_analyze.py").read_text(encoding="utf-8")
V = {"__name__": "v5b"}; exec(_src[:_src.index("out = {}; contrasts")], V)       # load (with A2), pm, summ, holm, ...
load, pm, summ, holm, signflip = V["load"], V["pm"], V["summ"], V["holm"], V["signflip"]
nums_clean, structural, verbal, digit_disc = V["nums_clean"], V["structural"], V["verbal"], V["digit_disclosure"]

args = sys.argv[1:]
if "--v5" in args:
    i = args.index("--v5"); runs, v5runs = args[:i], args[i + 1:]
else:
    runs = args; v5runs = [str(ROOT / "results" / ("pilot_v5_" + Path(r).name.replace("pilot_v5b_", ""))) for r in runs]

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def acc_pair(d, cell):
    """Per-pair one-answer accuracy (ties = 1/2), disclosing answers excluded."""
    u = d[d.use & (d.cell == cell)].assign(hit=lambda z: (z.llr > 0) + 0.5 * (z.llr == 0))
    return 100 * u.groupby("pair_id").hit.mean()
def same_answer(d, cell):
    a = d[d.cell == cell].assign(f=lambda z: z.nums.map(tuple))
    pv = a.pivot_table(index=["pair_id", "replicate"], columns="member", values="f", aggfunc="first")
    return int((pv.a == pv.b).sum()), int(len(pv))
def wilson_upper(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
def greedy_tables(run, instrs):
    G1 = {r["key"]: r for r in jl(Path(run) / "greedy_single.jsonl")}; G2 = {r["key"]: r for r in jl(Path(run) / "greedy_batched.jsonl")}
    cands = sorted({r["value"] for r in G1.values()})
    out = {}
    for instr in instrs:
        ref = {v: tuple(nums_clean(G1[f"{instr}|{v}"]["response"])) for v in cands}
        obs = {v: tuple(nums_clean(G2[f"{instr}|{v}"]["response"])) for v in cands}
        txt = {v: G2[f"{instr}|{v}"]["response"] for v in cands}
        disc = {v: structural(txt[v], v) or verbal(txt[v], v) or digit_disc(txt[v], v) for v in cands}
        groups = {}
        for v in cands: groups.setdefault(ref[v], []).append(v)
        def score(o, v): m = groups.get(o, []); return 1 / len(m) if v in m else 0.0
        clean = [v for v in cands if not disc[v]]
        cross = float(np.mean([score(obs[v], v) for v in clean])) if clean else float("nan")
        rng = np.random.default_rng(0); null = []
        vals = [obs[v] for v in clean]
        for _ in range(20000):
            perm = rng.permutation(len(vals))
            null.append(np.mean([score(vals[perm[k]], v) for k, v in enumerate(clean)]))
        p = (np.sum(np.array(null) >= cross) + 1) / 20001
        probs = np.array([len(g) for g in groups.values()]) / len(cands)
        out[instr] = dict(cross_setup=100 * cross, p=float(p), chance=100 / len(cands), n=len(clean), disclosing=int(sum(disc.values())),
                          cross_setup_all=100 * float(np.mean([score(obs[v], v) for v in cands])),
                          same_setup=100 * float(np.mean([1 / len(groups[ref[v]]) for v in cands])),
                          bits=max(0.0, float(-(probs * np.log2(probs)).sum())), max_bits=float(np.log2(len(cands))),
                          distinct_fingerprints=len(groups), batched_identical=100 * float(np.mean([obs[v] == ref[v] for v in cands])))
    return out

res, primary = {}, []
for run, v5run in zip(runs, v5runs):
    name = Path(run).name.replace("pilot_v5b_", "")
    d, d5 = load(run), load(v5run)
    instrs = sorted(d.instr.unique(), key=["none", "ord", "emph", "terse"].index)
    R = {}
    k, n = same_answer(d, "ord|T0.1"); R["B1"] = dict(same=k, n=n, rate=100 * k / max(n, 1), wilson_upper=100 * wilson_upper(k, n))
    R["B1"]["verdict"] = "supported (no collapse)" if R["B1"]["wilson_upper"] < 10 else "not supported"
    diff = (acc_pair(d, "ord|T0.1") - acc_pair(d5, "ord|T1.0")).dropna()
    R["B2"] = summ(diff); R["B2"]["acc_T01"] = float(acc_pair(d, "ord|T0.1").mean()); R["B2"]["acc_T1_v5"] = float(acc_pair(d5, "ord|T1.0").mean())
    G = greedy_tables(run, instrs); R["greedy"] = G; R["B3"] = G["ord"]
    primary += [(name, "B2", R["B2"]["p"], np.sign(R["B2"]["estimate"]) > 0), (name, "B3", R["B3"]["p"], R["B3"]["cross_setup"] > R["B3"]["chance"])]
    cells = []
    for c in sorted(d.cell.unique()):
        a = d[d.cell == c]; ks, ns = same_answer(d, c)
        cells.append(dict(cell=c, acc=float(acc_pair(d, c).mean()), same_answer=100 * ks / max(ns, 1), kl=float(pm(d[d.use & (d.cell == c)], "llr").mean()),
                          has_secret=100 * a.has_s.mean(), has_decoy=100 * a.has_d.mean(), verbal=100 * a.verbal.mean(), digit=100 * a.digit.mean()))
    R["cells"] = cells
    hit = [c for c in cells if c["same_answer"] >= 10 and c["cell"].endswith("T0.1")]
    if hit:
        e01 = (acc_pair(d, "emph|T0.1") - acc_pair(d, "ord|T0.1")); e1 = (acc_pair(d5, "emph|T1.0") - acc_pair(d5, "ord|T1.0"))
        R["B5"] = summ((e01 - e1).dropna()); R["B5"]["triggered_by"] = [c["cell"] for c in hit]
    else:
        R["B5"] = "not run: no cell at T = 0.1 reached a 10% same-answer rate"
    res[name] = R
adj = holm([p for _, _, p, _ in primary])
for (name, h, p, right), pa in zip(primary, adj):
    res[name][h]["p_holm"] = pa
    res[name][h]["verdict"] = "confirmed" if pa < 0.05 and right else ("opposite direction" if pa < 0.05 else "not supported")

for name, R in res.items():
    print(f"\n######## {name}")
    b1 = R["B1"]; print(f"  B1 same-answer, ord T=0.1: {b1['same']}/{b1['n']} = {b1['rate']:.1f}% (95% upper {b1['wilson_upper']:.1f}%) -> {b1['verdict']}")
    b2 = R["B2"]; print(f"  B2 one-answer accuracy ord: T=0.1 {b2['acc_T01']:.1f}% vs T=1.0 (v5) {b2['acc_T1_v5']:.1f}%: diff {b2['estimate']:+.2f} pp "
                        f"({b2['pairs_pos']}/{b2['n']} pairs > 0; LOO {b2['loo_min']:+.2f}..{b2['loo_max']:+.2f}) p = {b2['p']:.4f}, Holm {b2['p_holm']:.4f} -> {b2['verdict']}")
    b3 = R["B3"]; print(f"  B3 greedy cross-setup identification, ord: {b3['cross_setup']:.1f}% (chance {b3['chance']:.2f}%; n = {b3['n']}; "
                        f"p = {b3['p']:.4f}, Holm {b3['p_holm']:.4f}) -> {b3['verdict']}")
    print("  B4 greedy by instruction:")
    for i, gi in R["greedy"].items():
        print(f"    {i:6s} cross-setup {gi['cross_setup']:5.1f}% (all answers {gi['cross_setup_all']:5.1f}%) | same-setup {gi['same_setup']:5.1f}% | "
              f"bits {gi['bits']:.2f}/{gi['max_bits']:.2f} | fingerprints {gi['distinct_fingerprints']} | batched = single {gi['batched_identical']:.0f}% | disclosing {gi['disclosing']}")
    print("  B5:", R["B5"] if isinstance(R["B5"], str) else f"{R['B5']['estimate']:+.2f} pp (p = {R['B5']['p']:.4f})")
    print("  cells (B6 = has_secret / has_decoy):")
    print(pd.DataFrame(R["cells"]).round(2).to_string(index=False))
outp = (ROOT / "results" / "v5b_analysis.json") if all("results" in Path(r).parts for r in runs) else Path(runs[0]).parent / "v5b_analysis.json"
outp.write_text(json.dumps(res, indent=2, default=float)); print("\nwrote", outp)
