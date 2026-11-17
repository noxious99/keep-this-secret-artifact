"""Integrity check of the v7 full runs (preregistration_v7.md §6.4 step 3), before the pre-registered analysis.
Per model: pinned revision and transformers version; prompt hash; complete, no GPU error, not smoke/pilot; the exact pre-registered job and
greedy-key sets (rebuilt from the metadata's design lists and checked against §3.4); every answer scored, finite; no duplicates; the same
values in every model; watch rates (truncation, disclosure, exact value, fewer than 10 numbers, refusals-like answers, single = batched).
Also: the copy under results/attraction_vs_suppression/ (if any) is byte-identical.
Usage: python analysis/v7_integrity.py   -> results/v7_integrity.json"""
import json, sys, filecmp, math
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_analyze import jl, stratum4, nums4
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
HASH = "5f62a0b3b4cf"
PIN = {"phi-4": ("microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25", "5.21.1"),
       "falcon3-10b-instruct": ("tiiuae/Falcon3-10B-Instruct", "8799bc6aec0152757221dc6b272d824642db6202", "5.21.0"),
       "granite-33-8b-instruct": ("ibm-granite/granite-3.3-8b-instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b", "5.21.0"),
       "olmo-2-0325-32b-instruct": ("allenai/OLMo-2-0325-32B-Instruct", "b96024342a77a69aa0dda815c3454a671f477463", "5.21.0"),
       "llama-31-8b-instruct": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659", "5.21.0")}
FILES = ["metadata.json", "responses.jsonl", "scores.jsonl", "greedy_single.jsonl", "greedy_batched.jsonl"]

def expected(meta):
    lab = [tuple(x) for x in meta["labels"]]; labels = [x for x in lab if x[1] != "bridge"]; P = {p["pair_id"]: p for p in meta["pairs"]}
    lp, bp, lc, bc = meta["label_pairs"], meta["bridge_pairs"], meta["label_candidates"], meta["bridge_candidates"]
    jobs = {f"{l}|{p}|{m}|{c}|r0" for l, _ in labels for p in lp for m in "ab" for c in ("none", "instr")}
    jobs |= {f"secret|{p}|{m}|none|r0" for p in bp for m in "ab"}
    jobs |= {f"reference number|{p}|{m}|none|r0" for p in bp if p not in lp for m in "ab"}
    keys = {f"{l}|{c}|{v}" for l, _ in labels for c in ("none", "instr") for v in lc}
    keys |= {f"secret|none|{v}" for v in bc} | {f"reference number|none|{v}" for v in bc if v not in lc}
    return jobs, keys

if __name__ == "__main__":
    out, ok_all, designs = {}, True, {}
    for tag, (mid, rev, tf) in PIN.items():
        d = RES / f"v7_{tag}"; issues = []
        if not (d / "metadata.json").exists(): print(f"== {tag}: MISSING"); ok_all = False; continue
        m = json.load(open(d / "metadata.json", encoding="utf-8"))
        for k, want in [("model_id", mid), ("revision_pinned", rev), ("transformers", tf), ("prompt_hash", HASH), ("run_complete", True),
                        ("gpu_error", None), ("smoke_test", False), ("pilot", False)]:
            if m.get(k) != want: issues.append(f"{k} = {m.get(k)!r}, expected {want!r}")
        R = jl(d / "responses.jsonl"); S = jl(d / "scores.jsonl"); GS = jl(d / "greedy_single.jsonl"); GB = jl(d / "greedy_batched.jsonl")
        jobs, keys = expected(m)
        if len(jobs) != 2448 or len(keys) != 2040: issues.append(f"design size {len(jobs)}/{len(keys)} != 2448/2040")
        for name, rows, idk, want in [("responses", R, "job_id", jobs), ("scores", S, "job_id", jobs), ("greedy_single", GS, "key", keys), ("greedy_batched", GB, "key", keys)]:
            ids = [r[idk] for r in rows]; dup = [k for k, n in Counter(ids).items() if n > 1]
            if dup: issues.append(f"{name}: {len(dup)} duplicated ids")
            if set(ids) != want: issues.append(f"{name}: {len(want - set(ids))} missing, {len(set(ids) - want)} unexpected")
        sc = {s["job_id"]: s for s in S}
        bad = [k for k, s in sc.items() if not (math.isfinite(s["lp_value"]) and math.isfinite(s["lp_alt"]))]
        if bad: issues.append(f"{len(bad)} non-finite scores")
        designs[tag] = (json.dumps(m["pairs"]), json.dumps(m["candidates"]), json.dumps(m["labels"]))
        dup_copy = RES / "attraction_vs_suppression" / d.name
        copy_same = all(filecmp.cmp(d / f, dup_copy / f, shallow=False) for f in FILES) if dup_copy.exists() else None
        st = Counter(stratum4(r["response"], r["value"]) for r in R); n = len(R)
        words = sum(1 for r in R if len(nums4(r["response"])) < 10)
        Bk ={b["key"]: b["response"] for b in GB}; gsame = sum(1 for g in GS if Bk.get(g["key"]) == g["response"])
        by_cls = {}
        for r in R:
            k = f"{r['cls']}|{r['cond']}"; x = by_cls.setdefault(k, Counter()); x["n"] += 1; s4 = stratum4(r["response"], r["value"])
            x["disclose"] += s4 == 1; x["exact"] += s4 == 2; x["lt10"] += len(nums4(r["response"])) < 10; x["trunc"] += r["truncated"]
        rec = dict(issues=issues, copy_identical=copy_same, n=n, truncated=sum(r["truncated"] for r in R) / n, strata={str(k): v for k, v in sorted(st.items())},
                   fewer_than_10_numbers=words / n, greedy_truncated=(sum(g["truncated"] for g in GS) + sum(g["truncated"] for g in GB)) / (len(GS) + len(GB)),
                   single_equals_batched=gsame / len(GS), elapsed_h=m.get("elapsed_h"), times_s=m.get("times_s"),
                   by_class_cond={k: {kk: (v / x["n"] if kk != "n" else v) for kk, v in x.items()} for k, x in by_cls.items()})
        out[tag] = rec; ok_all &= not issues
        print(f"== {tag}: {'PASS' if not issues else 'FAIL'} {issues if issues else ''}")
        print(f"   {n} answers | truncated {rec['truncated']:.1%} | strata {rec['strata']} | <10 numbers {rec['fewer_than_10_numbers']:.1%} | "
              f"greedy truncated {rec['greedy_truncated']:.1%} | single = batched {rec['single_equals_batched']:.1%} | {rec['elapsed_h']} h | copy identical {copy_same}")
        for k in sorted(rec["by_class_cond"]):
            v = rec["by_class_cond"][k]; print(f"     {k:16s} n {v['n']:4d} | disclose {v['disclose']:.1%} | exact {v['exact']:.1%} | <10 numbers {v['lt10']:.1%} | truncated {v['trunc']:.1%}")
    same = len(set(designs.values())) == 1
    print("same pairs, candidates and labels in every model:", same); ok_all &= same
    out["all_pass"] = bool(ok_all); out["same_design_all_models"] = same
    (RES / "v7_integrity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ALL PASS" if ok_all else "SOME CHECKS FAIL"); sys.exit(0 if ok_all else 1)
