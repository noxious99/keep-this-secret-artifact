"""Shared helpers for the R0 reanalysis (analysis/r0_plan.md). Exploratory; imports the pre-registered parsing and
statistics from analysis/v5_analyze.py exactly as v6_analyze.py does, and adds the exact-occurrence strata of r0_plan §3."""
import hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
_src = (ROOT / "analysis/v5_analyze.py").read_text(encoding="utf-8")
V = {"__name__": "r0"}; exec(_src[:_src.index("out = {}; contrasts")], V)
load5, pm, summ, holm, signflip = V["load"], V["pm"], V["summ"], V["holm"], V["signflip"]
nums_clean, structural, verbal, digit_disc = V["nums_clean"], V["structural"], V["verbal"], V["digit_disclosure"]

# name: (sampled-run folder, version, greedy-run folder)
RUNS = {"Qwen3-32B": ("pilot_v5_qwen3-32b", "v5", "pilot_v5b_qwen3-32b"),
        "Mistral-Small-24B": ("pilot_v5_mistral-small-24b-instruct-2501", "v5", "pilot_v5b_mistral-small-24b-instruct-2501"),
        "Phi-4": ("pilot_v6_phi-4", "v6", "pilot_v6_phi-4"),
        "Falcon3-10B": ("pilot_v6_falcon3-10b-instruct", "v6", "pilot_v6_falcon3-10b-instruct"),
        "Granite-3.3-8B": ("pilot_v6_granite-33-8b-instruct", "v6", "pilot_v6_granite-33-8b-instruct"),
        "OLMo-2-32B": ("pilot_v6_olmo-2-0325-32b-instruct", "v6", "pilot_v6_olmo-2-0325-32b-instruct"),
        "Llama-3.1-8B": ("pilot_v6_llama-31-8b-instruct", "v6", "pilot_v6_llama-31-8b-instruct")}
V6_KEY = {"Phi-4": "phi-4", "Falcon3-10B": "falcon3-10b-instruct", "Granite-3.3-8B": "granite-33-8b-instruct",
          "OLMo-2-32B": "olmo-2-0325-32b-instruct", "Llama-3.1-8B": "llama-31-8b-instruct"}
V5_KEY = {"Qwen3-32B": "qwen3-32b", "Mistral-Small-24B": "mistral-small-24b-instruct-2501"}

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def cell(ver, instr, T=1.0, task="list"): return f"{instr}|T{T}" if ver == "v5" else f"{task}|{instr}"

def partial_occ(t, v, nums):
    """Stratum 3 (r0_plan §3): v inside a longer digit run, digit-reversed v in the list, or two consecutive
    single-digit runs spelling v."""
    runs = re.findall(r"\d+", t)
    if any(len(r) > 2 and v in r for r in runs): return True
    if v[::-1] in nums: return True
    return any(len(runs[i]) == 1 and len(runs[i + 1]) == 1 and runs[i] + runs[i + 1] == v for i in range(len(runs) - 1))

def stratum(t, v, nums):
    if structural(t, v) or verbal(t, v) or digit_disc(t, v): return 1
    if v in nums: return 2
    if partial_occ(t, v, nums): return 3
    return 4

def load(name):
    folder, ver, _ = RUNS[name]
    d = load5(RES / folder)
    if ver == "v6": d["cell"] = d.task + "|" + d.instr
    d["n_nums"] = d.nums.map(len); d["refusal"] = d.n_nums < 10
    d["stratum"] = [stratum(t, v, n) for t, v, n in zip(d.response, d.value, d.nums)]
    d["stratum_alt"] = [stratum(t, a, n) for t, a, n in zip(d.response, d.alt, d.nums)]   # relative to the decoy
    d["clean_alt"] = d.stratum_alt != 1                                                    # does not disclose the decoy
    return d, ver

def boot_ci(x, levels=(2.5, 97.5), n=10000, seed=0):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 2: return [float("nan")] * len(levels)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(1), levels)]

def spearman(x, y):
    x, y = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(x, y)[0, 1])

def chi2_sf_even(x, df):
    """Chi-square survival function for even df (closed form)."""
    assert df % 2 == 0 and df > 0
    h = x / 2.0; term = 1.0; s = 1.0
    for j in range(1, df // 2):
        term *= h / j; s += term
    return float(np.exp(-h) * s)

def greedy_scores(gfolder, instr):
    """Per-candidate replay score, the pre-registered rule (v6_analyze.greedy / v5b): references = single greedy answers,
    observed = batched greedy answer; candidates whose observed answer discloses are excluded."""
    G1 = {r["key"]: r for r in jl(RES / gfolder / "greedy_single.jsonl")}
    G2 = {r["key"]: r for r in jl(RES / gfolder / "greedy_batched.jsonl")}
    cands = sorted({r["value"] for r in G1.values()})
    ref = {v: tuple(nums_clean(G1[f"{instr}|{v}"]["response"])) for v in cands}
    groups = {}
    for v in cands: groups.setdefault(ref[v], []).append(v)
    out = {}
    for v in cands:
        t = G2[f"{instr}|{v}"]["response"]
        if structural(t, v) or verbal(t, v) or digit_disc(t, v): continue
        m = groups.get(tuple(nums_clean(t)), []); out[v] = 1 / len(m) if v in m else 0.0
    return out, len(cands)

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
