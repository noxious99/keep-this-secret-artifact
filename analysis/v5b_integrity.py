"""Strict integrity check of v5b runs against the job lists rebuilt from pilot_v5b_lowtemp.ipynb.
Usage: python analysis/v5b_integrity.py results/pilot_v5b_qwen3-32b results/pilot_v5b_mistral-small-24b-instruct-2501"""
import json, sys, math, collections, zipfile, hashlib, io, contextlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
nb = json.load(open(ROOT / "pilot_v5b_lowtemp.ipynb", encoding="utf-8"))
cells = {}
for c in nb["cells"]:
    if c["cell_type"] == "code":
        s = "".join(c["source"]); cells[s.split("\n")[0].split("cell: ")[1].rstrip("]")] = s
g = {}
exec("import os, json, time, random, re, hashlib\nfrom pathlib import Path", g)
g.update(SEED=20261027, PAIR_SEED=20261026, SMOKE_TEST=False, N_PAIRS=36, N_REPLICATES=3)
with contextlib.redirect_stdout(io.StringIO()):
    exec(cells["dataset"], g); exec(cells["prompts"], g)
gen = cells["generate"]; exec(gen[gen.index("def build_jobs"):gen.index("def cut_at_eos")], g)
JOBS = {j["job_id"]: j for j in g["build_jobs"]()}
GKEYS = {f"{i}|{v}" for i in g["INSTR"] for v in g["CANDIDATES"]}
PAIRS = {p["pair_id"]: (p["a"], p["b"]) for p in g["pairs"]}
print("expected: sampled", len(JOBS), "| greedy", len(GKEYS), "x 2 | candidates", len(g["CANDIDATES"]), "| hash", g["PROMPT_HASH"])

def rd(p):
    out, bad = [], 0
    for l in open(p, encoding="utf-8"):
        if l.strip():
            try: out.append(json.loads(l))
            except Exception: bad += 1
    return out, bad

for run in map(Path, sys.argv[1:]):
    print(f"\n==================== {run.name}")
    m = json.load(open(run / "metadata.json")); issues = []
    print({k: m.get(k) for k in ["model_id", "prompt_hash", "smoke_test", "seed", "pair_seed", "weights_source", "load_from", "devices",
                                 "n_responses", "run_complete", "elapsed_h", "greedy_determinism"]})
    if m["prompt_hash"] != g["PROMPT_HASH"]: issues.append("hash")
    if m["smoke_test"] or not m["run_complete"]: issues.append("smoke flag or incomplete")
    if {p["pair_id"]: (p["a"], p["b"]) for p in m["pairs"]} != PAIRS: issues.append("pairs")
    if m["candidates"] != g["CANDIDATES"]: issues.append("candidates")
    R, bR = rd(run / "responses.jsonl"); S, bS = rd(run / "scores.jsonl")
    ids = [r["job_id"] for r in R]; dup = sum(v > 1 for v in collections.Counter(ids).values())
    miss, extra = set(JOBS) - set(ids), set(ids) - set(JOBS)
    fields = sum(1 for r in R if r["job_id"] in JOBS and any(str(r[k]) != str(JOBS[r["job_id"]][k]) for k in ["value", "alt", "instr", "temp", "replicate"]))
    smiss = set(ids) - {s["job_id"] for s in S}
    nonfin = sum(1 for s in S if not (math.isfinite(s["lp_value"]) and math.isfinite(s["lp_alt"])))
    print(f"sampled {len(R)} (bad {bR}, dup {dup}, missing {len(miss)}, unexpected {len(extra)}, field mismatch {fields}) | "
          f"scores {len(S)} (bad {bS}, missing {len(smiss)}, non-finite {nonfin})")
    if bR or dup or miss or extra or fields or bS or smiss or nonfin: issues.append("sampled")
    for name in ["greedy_single.jsonl", "greedy_batched.jsonl"]:
        G, bG = rd(run / name); keys = [x["key"] for x in G]
        gd = sum(v > 1 for v in collections.Counter(keys).values()); gm, ge = GKEYS - set(keys), set(keys) - GKEYS
        gfield = sum(1 for x in G if x["key"] != f"{x['instr']}|{x['value']}")
        print(f"{name}: {len(G)} (bad {bG}, dup {gd}, missing {len(gm)}, unexpected {len(ge)}, key/field mismatch {gfield}, "
              f"truncated {sum(x['truncated'] for x in G)}, empty {sum(not x['response'].strip() for x in G)})")
        if bG or gd or gm or ge or gfield: issues.append(name)
    df = pd.DataFrame(R)
    print("per cell:", sorted(set(df.groupby(["instr", "temp"]).size())), "| truncated", int(df.truncated.sum()),
          "| <think>", int(df.response.str.contains("<think>|</think>").sum()), "| tokens mean/max", round(df.n_tokens.mean()), df.n_tokens.max())
    z = next((p for p in run.parent.glob("*.zip") if any(i.filename.endswith(f"{run.name}/responses.jsonl") for i in zipfile.ZipFile(p).infolist())), None)
    if z:
        zz = zipfile.ZipFile(z)
        same = all(hashlib.md5(zz.read(i)).hexdigest() == hashlib.md5((run / Path(i.filename).name).read_bytes()).hexdigest()
                   for i in zz.infolist() if f"{run.name}/" in i.filename and not i.filename.endswith("/"))
        print("zip:", z.name, "identical to folder:", same)
        if not same: issues.append("zip differs")
    print("ISSUES:", issues or "none")
