import json, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "analysis/v5_analyze.py").read_text(encoding="utf-8")
V = {"__name__": "x"}; exec(src[:src.index("out = {}; contrasts")], V)
load, pm, summ = V["load"], V["pm"], V["summ"]
O = json.load(open(ROOT / "results/v6_analysis.json", encoding="utf-8")); B = json.load(open(ROOT / "results/v5b_analysis.json"))
V6 = {"Phi-4": "phi-4", "Falcon3-10B": "falcon3-10b-instruct", "Granite-3.3-8B": "granite-33-8b-instruct",
      "OLMo-2-32B": "olmo-2-0325-32b-instruct", "Llama-3.1-8B": "llama-31-8b-instruct"}
print("=== black-box replay identification (%) by instruction, list task, greedy (chance 1.25%)")
rows = {"Qwen3-32B (v5b)": {i: B["qwen3-32b"]["greedy"][i]["cross_setup"] for i in ["none", "ord", "emph", "terse"]},
        "Mistral-24B (v5b)": {i: B["mistral-small-24b-instruct-2501"]["greedy"][i]["cross_setup"] for i in ["none", "ord", "emph", "terse"]}}
for n, k in V6.items(): rows[n] = {i: O["per_model"][k]["greedy"][i]["replay"] for i in ["none", "ref", "ord", "emph", "terse"]}
print(pd.DataFrame(rows).T[["none", "ref", "ord", "emph", "terse"]].round(1).to_string())

def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]] exact two-sided p."""
    n1, n2, k = a + b, c + d, a + c; N = n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(N, k)
    p0 = pr(a); return min(1.0, sum(pr(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pr(x) <= p0 * (1 + 1e-9)))
print("\n=== the instruction makes the model state the secret: verbal disclosures, no instruction vs verbose policy (list task)")
RUNS = {"Qwen3-32B (v5)": ("pilot_v5_qwen3-32b", "v5"), "Mistral-24B (v5)": ("pilot_v5_mistral-small-24b-instruct-2501", "v5"),
        **{n: (f"pilot_v6_{k}", "v6") for n, k in V6.items()}}
tot = {"none": [0, 0], "ord": [0, 0], "emph": [0, 0]}
for name, (folder, ver) in RUNS.items():
    d = load(ROOT / "results" / folder)
    if ver == "v6": d["cell"] = d.task + "|" + d.instr
    c = (lambda i: f"{i}|T1.0") if ver == "v5" else (lambda i: f"list|{i}")
    disc = lambda i: d[d.cell == c(i)].pipe(lambda a: (int((a.verbal | a.structural | a.digit).sum()), len(a)))
    out = {i: disc(i) for i in ["none", "ord", "emph"]}
    for i in out: tot[i][0] += out[i][0]; tot[i][1] += out[i][1]
    pe = fisher_two_sided(out["emph"][0], out["emph"][1] - out["emph"][0], out["none"][0], out["none"][1] - out["none"][0])
    po = fisher_two_sided(out["ord"][0], out["ord"][1] - out["ord"][0], out["none"][0], out["none"][1] - out["none"][0])
    print(f"  {name:18s} none {out['none'][0]:3d}/{out['none'][1]} | ord {out['ord'][0]:3d}/{out['ord'][1]} (p {po:.4f}) | emph {out['emph'][0]:3d}/{out['emph'][1]} (p {pe:.2g})")
print("  pooled:", {i: f"{v[0]}/{v[1]} = {100 * v[0] / v[1]:.1f}%" for i, v in tot.items()})

print("\n=== pre-registered robustness for P1 / P2: all answers included, median, leave-one-pair-out")
for name, (folder, ver) in RUNS.items():
    d = load(ROOT / "results" / folder)
    if ver == "v6": d["cell"] = d.task + "|" + d.instr
    c = (lambda i: f"{i}|T1.0") if ver == "v5" else (lambda i: f"list|{i}")
    for lab, a_, b_ in [("P1 ord−none", "ord", "none"), ("P2 emph−ord", "emph", "ord")]:
        sp = summ((pm(d[d.use & (d.cell == c(a_))], "llr") - pm(d[d.use & (d.cell == c(b_))], "llr")).dropna())
        sa = summ((pm(d[d.cell == c(a_)], "llr") - pm(d[d.cell == c(b_)], "llr")).dropna())
        print(f"  {name:18s} {lab}: primary {sp['estimate']:+.2f} (median {sp['median']:+.2f}, LOO {sp['loo_min']:+.2f}..{sp['loo_max']:+.2f}) | all answers {sa['estimate']:+.2f} (p {sa['p']:.4f})")
