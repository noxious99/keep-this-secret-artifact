"""Exploratory: every angle on v5 + v6 (7 models), list task, T = 1. Post hoc; nothing here changes a pre-registered verdict."""
import json, re, collections, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "analysis/v5_analyze.py").read_text(encoding="utf-8")
V = {"__name__": "x"}; exec(src[:src.index("out = {}; contrasts")], V)
load, pm, summ = V["load"], V["pm"], V["summ"]

RUNS = {"Qwen3-32B (v5)": ("pilot_v5_qwen3-32b", "v5"), "Mistral-24B (v5)": ("pilot_v5_mistral-small-24b-instruct-2501", "v5"),
        "Phi-4": ("pilot_v6_phi-4", "v6"), "Falcon3-10B": ("pilot_v6_falcon3-10b-instruct", "v6"),
        "Granite-3.3-8B": ("pilot_v6_granite-33-8b-instruct", "v6"), "OLMo-2-32B": ("pilot_v6_olmo-2-0325-32b-instruct", "v6"),
        "Llama-3.1-8B": ("pilot_v6_llama-31-8b-instruct", "v6")}
def cellname(ver, instr, task="list"): return f"{instr}|T1.0" if ver == "v5" else f"{task}|{instr}"
rows = []; D = {}
for name, (folder, ver) in RUNS.items():
    d = load(ROOT / "results" / folder)
    if ver == "v6": d["cell"] = d.task + "|" + d.instr
    d["nn"] = d.nums.map(len); d["refusal"] = d.nn < 10
    D[name] = (d, ver)
    u = d[d.use]; kl = lambda c, u=u: pm(u[u.cell == c], "llr"); gap = lambda c, d=d: 100 * (pm(d[d.cell == c], "has_s") - pm(d[d.cell == c], "has_d"))
    C = lambda i, ver=ver: cellname(ver, i)
    r = dict(model=name)
    for i in ["none", "ord", "emph", "terse"] + (["ref"] if ver == "v6" else []):
        a = d[d.cell == C(i)]
        r[f"KL_{i}"] = kl(C(i)).mean(); r[f"gap_{i}"] = gap(C(i)).mean()
        r[f"verbal_{i}"] = 100 * a.verbal.mean(); r[f"struct_{i}"] = 100 * a.structural.mean(); r[f"refusal_{i}"] = 100 * a.refusal.mean()
    for a_, b_ in [("ord", "none"), ("emph", "ord"), ("terse", "none"), ("emph", "none")] + ([("ref", "none")] if ver == "v6" else []):
        s = summ((kl(C(a_)) - kl(C(b_))).dropna()); r[f"d_{a_}_{b_}"] = s["estimate"]; r[f"p_{a_}_{b_}"] = s["p"]; r[f"pos_{a_}_{b_}"] = f"{s['pairs_pos']}/{s['n']}"
    rows.append(r)
T = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("=== KL per instruction (nats/answer, list task, T=1)")
print(T[[c for c in T if c.startswith("KL_")]].round(2).to_string())
print("\n=== contrasts (estimate, pairs>0, p)")
for a_, b_ in [("ord", "none"), ("emph", "ord"), ("terse", "none"), ("emph", "none"), ("ref", "none")]:
    k = f"d_{a_}_{b_}"
    print(f"  {a_} − {b_}:", {m: (round(T.loc[m, k], 2), T.loc[m, f'pos_{a_}_{b_}'], round(T.loc[m, f'p_{a_}_{b_}'], 4)) for m in T.index if pd.notna(T.loc[m].get(k))})
print("\n=== containment gap (secret − decoy lists, pp)")
print(T[[c for c in T if c.startswith("gap_")]].round(1).to_string())
print("\n=== verbal disclosure % per instruction")
print(T[[c for c in T if c.startswith("verbal_")]].round(1).to_string())
print("\n=== refusal % (< 10 numbers) per instruction")
print(T[[c for c in T if c.startswith("refusal_")]].round(1).to_string())
# baseline leakage vs instruction effect
x, y = T.KL_none, T.d_ord_none
rho = x.rank().corr(y.rank()); n = len(x)
null = [x.rank().corr(pd.Series(np.array(p), index=x.index).rank()) for p in itertools.permutations(y.values)]
print(f"\n=== baseline KL(none) vs Δ(ord−none): Spearman {rho:+.2f}, exact two-sided p = {np.mean(np.abs(null) >= abs(rho) - 1e-12):.3f} (n = {n})")
x2 = T.gap_none
rho2 = x2.rank().corr(y.rank()); null2 = [x2.rank().corr(pd.Series(np.array(p), index=x2.index).rank()) for p in itertools.permutations(y.values)]
print(f"=== baseline gap(none) vs Δ(ord−none): Spearman {rho2:+.2f}, exact two-sided p = {np.mean(np.abs(null2) >= abs(rho2) - 1e-12):.3f}")
print(f"=== spread of KL across models: none SD {T.KL_none.std():.2f} (range {T.KL_none.min():.2f}–{T.KL_none.max():.2f}) | ord SD {T.KL_ord.std():.2f} (range {T.KL_ord.min():.2f}–{T.KL_ord.max():.2f})")
# v4 also had verbose-vs-default
print("\nv4 verbose − default (Fairoze cell): Qwen3-8B −3.10, Qwen3-14B +0.33, Qwen3-32B +2.56, Llama-3.1-8B −0.01")
