"""R0 nearest-reference attacker (analysis/r0_plan.md §9). Outcome-blind with respect to the nearest-reference metric
(exact-match results on the same answers were already seen). Exploratory; no pre-registered verdict changes.

Threat model: closed set of 80 known candidates; the attacker knows the model, the prompt format and the instruction,
generates one greedy single-call reference answer per candidate offline (80 model calls), and observes one SAMPLED answer.

Usage: python analysis/r0_nearest_ref.py   -> results/r0_nearest_ref.json (+ printed summary). Deterministic (seed 0)."""
import datetime, json, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r0_common import RES, jl, nums_clean, stratum, holm, signflip, boot_ci, sha256, jsonable
sys.stdout.reconfigure(encoding="utf-8")
try:
    from rapidfuzz.distance import Levenshtein as _LV
    def lev(a, b): return _LV.distance(a, b)
except ImportError:                                                     # pure-Python fallback, same result
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]

N_PERM = 20000; N_BOOT = 10000; SEED = 0

# ------------------------------------------------------------------ similarity metrics (higher = closer)
def sim_positional(o, r):                                               # PRIMARY (r0_plan §9)
    if not o and not r: return 0.0
    return sum(1 for i in range(min(len(o), len(r))) if o[i] == r[i]) / max(len(o), len(r))
def sim_prefix(o, r):
    k = 0
    for x, y in zip(o, r):
        if x != y: break
        k += 1
    return float(k)
def sim_jaccard(o, r):
    a, b = set(o), set(r); return len(a & b) / len(a | b) if (a | b) else 0.0
def sim_lev(o, r):
    m = max(len(o), len(r)); return 1.0 - lev(list(o), list(r)) / m if m else 0.0
def sim_length(o, r): return -abs(len(o) - len(r))
METRICS = {"nearest_positional": sim_positional, "prefix": sim_prefix, "jaccard": sim_jaccard, "levenshtein": sim_lev, "length_only": sim_length}

def best_set(o, refs, metric):
    """Indices of the candidates the attacker would pick (argmax with ties). 'exact' keeps the pre-registered rule:
    the candidates whose reference equals o (possibly none, which scores 0)."""
    if metric == "exact": return [k for k, r in enumerate(refs) if r == o]
    s = np.array([METRICS[metric](o, r) for r in refs]); return list(np.flatnonzero(s >= s.max() - 1e-12))

def group_eval(answers, cands, refs, metric, exclusion):
    """answers: list of dicts (value, nums, stratum). Returns per-secret matrix S[i, j] over the group's secrets and the
    secret list. S[i, j] = mean over secret i's retained answers of score(answer | truth = secrets[j])."""
    keep = [a for a in answers if (exclusion == "all") or (exclusion == "primary" and a["stratum"] != 1) or (exclusion == "s4" and a["stratum"] == 4)]
    secrets = sorted({a["value"] for a in keep}); si = {v: i for i, v in enumerate(secrets)}; ci = {v: k for k, v in enumerate(cands)}
    S = np.zeros((len(secrets), len(secrets))); cnt = np.zeros(len(secrets))
    for a in keep:
        b = best_set(tuple(a["nums"]), refs, metric); row = np.zeros(len(secrets))
        if b:
            bs = {cands[k] for k in b}
            for v in secrets:
                if v in bs: row[si[v]] = 1.0 / len(b)
        S[si[a["value"]]] += row; cnt[si[a["value"]]] += 1
    S = S / np.maximum(cnt, 1)[:, None]
    return S, secrets, len(keep)

def perm_test(S, rng):
    obs = float(np.mean(np.diag(S))); n = len(S)
    perms = np.argsort(rng.random((N_PERM, n)), axis=1)
    null = S[np.arange(n)[None, :], perms].mean(1)
    return obs, float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)), float(null.mean()), float(np.percentile(null, 95))

# ------------------------------------------------------------------ data
V5B = {"Qwen3-32B": "pilot_v5b_qwen3-32b", "Mistral-Small-24B": "pilot_v5b_mistral-small-24b-instruct-2501"}
V5 = {"Qwen3-32B": "pilot_v5_qwen3-32b", "Mistral-Small-24B": "pilot_v5_mistral-small-24b-instruct-2501"}
V6 = {"Phi-4": "pilot_v6_phi-4", "Falcon3-10B": "pilot_v6_falcon3-10b-instruct", "Granite-3.3-8B": "pilot_v6_granite-33-8b-instruct",
      "OLMo-2-32B": "pilot_v6_olmo-2-0325-32b-instruct", "Llama-3.1-8B": "pilot_v6_llama-31-8b-instruct"}
def refs_of(folder, instr):
    G1 = {r["key"]: r for r in jl(RES / folder / "greedy_single.jsonl")}
    cands = sorted({r["value"] for r in G1.values()})
    return cands, [tuple(nums_clean(G1[f"{instr}|{v}"]["response"])) for v in cands]
def answers_of(folder, instr, T, task=None):
    out = []
    for r in jl(RES / folder / "responses.jsonl"):
        if r["instr"] != instr or abs(r["temp"] - T) > 1e-9 or (task is not None and r.get("task") != task): continue
        n = nums_clean(r["response"]); out.append(dict(value=r["value"], nums=n, stratum=stratum(r["response"], r["value"], n), length=len(n)))
    return out

GROUPS = []   # (source, model, instr, T, answers folder, refs folder, task)
for m in V5B:
    for T in [0.1, 0.3]:
        for i in ["none", "ord", "emph", "terse"]: GROUPS.append(("v5b", m, i, T, V5B[m], V5B[m], None))
    for T in [1.0, 0.7, 0.5]:
        for i in ["none", "ord", "emph", "terse"]: GROUPS.append(("v5_vs_v5b_refs", m, i, T, V5[m], V5B[m], None))
for m in V6:
    for i in ["none", "ref", "ord", "emph", "terse"]: GROUPS.append(("v6", m, i, 1.0, V6[m], V6[m], "list"))
PRIMARY = {("v5b", m, "ord", T) for m in V5B for T in [0.1, 0.3]}

# ------------------------------------------------------------------ self-check: v5b_probe item 2 (dossier §18: 20.4% / 12.2%)
checks = []
for m, target in [("Qwen3-32B", 20.4), ("Mistral-Small-24B", 12.2)]:
    cands, refs = refs_of(V5B[m], "ord"); A = answers_of(V5B[m], "ord", 0.1)
    sc = []
    for a in A:
        b = best_set(tuple(a["nums"]), refs, "exact"); sc.append(1 / len(b) if b and a["value"] in {cands[k] for k in b} else 0.0)
    mine = 100 * float(np.mean(sc)); checks.append(dict(check=f"{m} exact identification, ord, T=0.1, all answers (v5b_probe item 2)",
                                                        mine=mine, reference=target, ok=bool(abs(mine - target) <= 0.05)))
print("SELF-CHECK:", [(c["check"][:30], round(c["mine"], 2), c["reference"], c["ok"]) for c in checks])
if not all(c["ok"] for c in checks):
    Path(RES / "r0_nearest_ref.json").write_text(json.dumps(jsonable(dict(self_check=checks)), indent=1)); sys.exit(1)

# ------------------------------------------------------------------ evaluate every group
rng = np.random.default_rng(SEED); results = []
for src, m, instr, T, af, rf, task in GROUPS:
    cands, refs = refs_of(rf, instr); A = answers_of(af, instr, T, task)
    if not A: continue
    g = dict(source=src, model=m, instr=instr, T=T, n_answers=len(A), n_candidates=len(cands), chance_pct=100 / len(cands),
             primary=(src, m, instr, T) in PRIMARY,
             strata={str(k): int(v) for k, v in sorted(Counter(a["stratum"] for a in A).items())},
             length_observed={str(k): int(v) for k, v in sorted(Counter(a["length"] for a in A).items())},
             length_references={str(k): int(v) for k, v in sorted(Counter(len(r) for r in refs).items())})
    per_secret = {}
    metrics = ["exact", "nearest_positional"] + (["prefix", "jaccard", "levenshtein", "length_only"] if src == "v5b" else ["length_only"])
    for metric in metrics:
        for excl in (["primary", "all", "s4"] if metric in ("exact", "nearest_positional") else ["primary"]):
            S, secrets, nkeep = group_eval(A, cands, refs, metric, excl)
            if not secrets: continue
            obs, p, nmean, n95 = perm_test(S, rng)
            g[f"{metric}|{excl}"] = dict(score_pct=100 * obs, p_perm=p, null_mean_pct=100 * nmean, null_p95_pct=100 * n95,
                                        n_secrets=len(secrets), n_answers=nkeep)
            if excl == "primary": per_secret[metric] = pd.Series(np.diag(S), index=secrets)
    d = (per_secret["nearest_positional"] - per_secret["exact"]).dropna()
    g["nearest_minus_exact|primary"] = dict(diff_pp=100 * float(d.mean()), ci95=boot_ci(100 * d, n=N_BOOT, seed=SEED),
                                            p_signflip_two_sided=float(signflip(d)) if np.any(d != 0) else 1.0, n_secrets=len(d))
    results.append(g)

# ------------------------------------------------------------------ Holm over the 4 primary tests
prim = [g for g in results if g["primary"]]
adj = holm([g["nearest_positional|primary"]["p_perm"] for g in prim])
for g, pa in zip(prim, adj): g["nearest_positional|primary"]["p_holm4"] = pa

def reading(g):
    a = g["nearest_positional|primary"]; dd = g["nearest_minus_exact|primary"]
    above = a.get("p_holm4", a["p_perm"]) < 0.05
    better = dd["ci95"][0] > 0 and dd["p_signflip_two_sided"] < 0.05
    if above and better: return "above chance; significantly better than exact matching"
    if above: return "above chance; improvement over exact matching inconclusive"
    return "not established as an effective attacker in this group"
for g in results: g["reading"] = reading(g)

files = sorted({f for _, _, _, _, af, rf, _ in GROUPS for f in (af, rf)})
OUT = dict(meta=dict(script="analysis/r0_nearest_ref.py", plan="analysis/r0_plan.md §9", date=str(datetime.date.today()), seed=SEED,
                     permutations=N_PERM, bootstrap=N_BOOT, edit_distance="rapidfuzz" if "_LV" in globals() else "python",
                     blinding="outcome-blind w.r.t. the nearest-reference metric; exact-match results previously seen",
                     threat_model="closed set of 80 candidates; attacker knows model, prompt format and instruction; 80 offline greedy "
                                  "single-call references (80 model calls); observes one sampled answer. Failure does not show that "
                                  "sampled answers are safe.",
                     notes=["v5 answers are matched against v5b references (same models and instruction texts, different run)",
                            "prefix, jaccard and levenshtein metrics computed for v5b groups only (descriptive)"],
                     inputs_sha256={f: {fn: sha256(RES / f / fn) for fn in ["responses.jsonl", "greedy_single.jsonl"]
                                        if (RES / f / fn).exists()} for f in files}),
           self_check=checks, primary_tests=[dict(model=g["model"], T=g["T"], **g["nearest_positional|primary"]) for g in prim],
           groups=results)
Path(RES / "r0_nearest_ref.json").write_text(json.dumps(jsonable(OUT), indent=1), encoding="utf-8")

# ------------------------------------------------------------------ printed summary
print("\n=== PRIMARY (v5b, ord): nearest-reference vs chance (Holm over 4) and vs exact")
for g in prim:
    a = g["nearest_positional|primary"]; e = g["exact|primary"]; dd = g["nearest_minus_exact|primary"]
    print(f"{g['model']:18s} T={g['T']}: nearest {a['score_pct']:5.1f}% (null mean {a['null_mean_pct']:.2f}%, p_holm {a['p_holm4']:.4f}) | "
          f"exact {e['score_pct']:5.1f}% | diff {dd['diff_pp']:+.1f} pp {[round(x, 1) for x in dd['ci95']]} p {dd['p_signflip_two_sided']:.4f} -> {g['reading']}")
print("\n=== all groups (primary exclusion): exact | nearest | length-only | (v5b: prefix, jaccard, levenshtein) | nearest on stratum 4")
for g in results:
    row = f"{g['source']:15s} {g['model']:18s} {g['instr']:5s} T={g['T']:<3}: exact {g['exact|primary']['score_pct']:5.1f} | nearest {g['nearest_positional|primary']['score_pct']:5.1f} (p {g['nearest_positional|primary']['p_perm']:.3f}) | len {g['length_only|primary']['score_pct']:4.1f}"
    if g["source"] == "v5b": row += f" | pre {g['prefix|primary']['score_pct']:5.1f} jac {g['jaccard|primary']['score_pct']:5.1f} lev {g['levenshtein|primary']['score_pct']:5.1f}"
    if "nearest_positional|s4" in g: row += f" | s4 {g['nearest_positional|s4']['score_pct']:5.1f}"
    print(row)
print("\nwrote", RES / "r0_nearest_ref.json")
