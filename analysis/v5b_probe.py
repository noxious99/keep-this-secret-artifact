"""Exploratory follow-ups to v5b (post hoc, 2026-10-26): attack assumptions and hidden gaps.
1. wrong-template attack: references generated under one instruction, observed answer under another
2. sampled low-temperature answers matched against greedy references (realistic low-T deployments)
3. what collided fingerprints still reveal (first digit, second digit)
4. priming at low temperature: containment gap per cell, pair-level tests
5. verbal disclosures under the verbose policy at low temperature (examples)
Usage: python analysis/v5b_probe.py results/pilot_v5b_qwen3-32b results/pilot_v5b_mistral-small-24b-instruct-2501"""
import json, re, sys, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_src = (ROOT / "analysis/v5_analyze.py").read_text(encoding="utf-8")
V = {"__name__": "probe"}; exec(_src[:_src.index("out = {}; contrasts")], V)
load, pm, summ, nums_clean = V["load"], V["pm"], V["summ"], V["nums_clean"]
def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
INSTR = ["none", "ord", "emph", "terse"]

for run in sys.argv[1:]:
    run = Path(run); name = run.name.replace("pilot_v5b_", "")
    G1 = {r["key"]: tuple(nums_clean(r["response"])) for r in jl(run / "greedy_single.jsonl")}
    G2 = {r["key"]: tuple(nums_clean(r["response"])) for r in jl(run / "greedy_batched.jsonl")}
    cands = sorted({k.split("|")[1] for k in G1})
    print(f"\n######## {name}")

    # 1. wrong-template attack (cross-setup: observed batched under `obs_i`, references single under `ref_i`)
    print("  1. identification (%) when the attacker's references use a different instruction (rows: true, cols: attacker's guess)")
    tab = pd.DataFrame(index=INSTR, columns=INSTR, dtype=float)
    for oi in INSTR:
        for ri in INSTR:
            groups = collections.defaultdict(list)
            for v in cands: groups[G1[f"{ri}|{v}"]].append(v)
            s = [1 / len(groups[G2[f"{oi}|{v}"]]) if v in groups.get(G2[f"{oi}|{v}"], []) else 0 for v in cands]
            tab.loc[oi, ri] = 100 * np.mean(s)
    print(tab.round(1).to_string())
    # attacker who does not know the instruction: union of references over all four instructions
    allref = collections.defaultdict(set)
    for ri in INSTR:
        for v in cands: allref[G1[f"{ri}|{v}"]].add(v)
    unk = {oi: 100 * np.mean([1 / len(allref[G2[f"{oi}|{v}"]]) if v in allref.get(G2[f"{oi}|{v}"], set()) else 0 for v in cands]) for oi in INSTR}
    print("     attacker tries all four instructions (references pooled):", {k: round(x, 1) for k, x in unk.items()})

    # 2. sampled low-temperature answers vs greedy references
    R = jl(run / "responses.jsonl")
    print("  2. sampled answers that exactly equal the greedy reference of their secret / identified among 80 via greedy references")
    for T in [0.3, 0.1]:
        row = {}
        for ins in INSTR:
            rs = [r for r in R if r["temp"] == T and r["instr"] == ins]
            groups = collections.defaultdict(list)
            for v in cands: groups[G1[f"{ins}|{v}"]].append(v)
            eq = np.mean([tuple(nums_clean(r["response"])) == G1[f"{ins}|{r['value']}"] for r in rs])
            ident = np.mean([1 / len(groups[f]) if r["value"] in groups.get(f, []) else 0 for r in rs for f in [tuple(nums_clean(r["response"]))]])
            reps = collections.defaultdict(set)
            for r in rs: reps[(r["pair_id"], r["member"])].add(tuple(nums_clean(r["response"])))
            row[ins] = f"equal {100*eq:.0f}%, identified {100*ident:.0f}%, replicates identical {100*np.mean([len(s) == 1 for s in reps.values()]):.0f}%"
        print(f"     T={T}:", row)

    # 3. what collided fingerprints still reveal
    print("  3. collisions: candidates sharing a greedy fingerprint")
    for ins in INSTR:
        groups = collections.defaultdict(list)
        for v in cands: groups[G1[f"{ins}|{v}"]].append(v)
        col = [g for g in groups.values() if len(g) > 1]
        same_first = np.mean([len({x[0] for x in g}) == 1 for g in col]) if col else float("nan")
        same_second = np.mean([len({x[1] for x in g}) == 1 for g in col]) if col else float("nan")
        biggest = max(col, key=len) if col else []
        print(f"     {ins:6s} {len(col)} groups, {sum(map(len, col))} candidates; same first digit in {100*same_first:.0f}% of groups, "
              f"same second digit in {100*same_second:.0f}%; largest {biggest}")

    # 4. priming at low temperature: containment gap per cell (pair-level), all temperatures incl. v5
    d = load(run)
    print("  4. containment gap (secret − decoy, pp), pair-level")
    for ins in INSTR:
        out = []
        for T in [0.3, 0.1]:
            c = f"{ins}|T{T}"; gap = 100 * (pm(d[d.cell == c], "has_s") - pm(d[d.cell == c], "has_d")); s = summ(gap)
            out.append(f"T={T}: {s['estimate']:+.1f} (p {s['p']:.3f}, {s['pairs_pos']}/{s['n']})")
        print(f"     {ins:6s}", " | ".join(out))

    # 5. verbal disclosures at low temperature
    vb = d[d.verbal]
    print(f"  5. verbal disclosures: {len(vb)} ({vb.groupby('cell').size().to_dict()})")
    for _, r in vb.head(3).iterrows():
        print("     ", r.job_id, "secret", r.value, "|", re.sub(r"\s+", " ", r.response)[:150])
    gd = [r for r in jl(run / "greedy_single.jsonl") if V["verbal"](r["response"], r["value"])]
    print(f"     greedy single answers stating the secret: {len(gd)} {[r['key'] for r in gd]}")
