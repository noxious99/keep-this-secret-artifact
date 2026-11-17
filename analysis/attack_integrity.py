"""Integrity check of one attacker-study run folder (results/attack_<tag>/) before it enters attack_analyze.py.
Checks completeness, duplicates and value ranges against the frozen design (preregistration_attack.md §4-§5); prints
descriptive watch rates (truncation, empty answers, occurrence strata). Computes no attacker accuracy and no hypothesis.

Usage: python analysis/attack_integrity.py results/attack_falcon3-10b-instruct [more folders ...]"""
import json, math, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_analyze import jl, CANDS, stratum4
from r0_common import nums_clean, stratum as stratum2
sys.stdout.reconfigure(encoding="utf-8")

HASH = "58b9866ce681"
CONDS_A, TEMPS_A, CONDS_B = ["none", "ord", "ref"], [1.0, 0.3], ["ord", "ref"]
N_REF, N_TGT, N_ANS_B, SPLITS = 20, 3, 10, {"train": 500, "val": 100, "test": 200}
N_CURVE, SEEDS_EXTRA = [50, 100, 200, 500], {"llama-31-8b-instruct": [1, 2]}

def check(d):
    d = Path(d); tag = d.name.removeprefix("attack_"); fails = []; ok = lambda c, msg: c or fails.append(msg)
    print(f"\n=== {d.name} ===")
    m = json.load(open(d / "metadata.json", encoding="utf-8"))
    ok(m["prompt_hash"] == HASH, f"prompt hash {m['prompt_hash']}"); ok(not m["smoke_test"], "smoke run")
    ok(all(m["stored_greedy_reproduces"].values()), "stored greedy does not reproduce")
    print("metadata:", dict(model=m["model_id"], revision=m["revision"], transformers=m["transformers"], run_complete=m["run_complete"],
                            elapsed_h=m["elapsed_h"], counts=m["counts"]))

    # Arm A sampled: every (cond, T, value) has 20 references and 3 targets, no duplicates
    A = jl(d / "armA_sampled.jsonl"); keys = Counter((r["role"], r["cond"], r["T"], r["value"], r["idx"]) for r in A)
    ok(max(keys.values()) == 1, f"Arm A duplicates: {sum(v > 1 for v in keys.values())}")
    exp = {(role, c, T, v, i) for c in CONDS_A for T in TEMPS_A for v in CANDS for role, n in (("ref", N_REF), ("target", N_TGT)) for i in range(n)}
    ok(set(keys) == exp, f"Arm A cells: {len(exp - set(keys))} missing, {len(set(keys) - exp)} unexpected")

    # Arm B sampled: 800 values, disjoint splits 500/100/200, 10 answers per value and condition
    B = jl(d / "armB_sampled.jsonl"); kb = Counter((r["cond"], r["value"], r["idx"]) for r in B)
    ok(max(kb.values()) == 1, f"Arm B duplicates: {sum(v > 1 for v in kb.values())}")
    split_of = {}
    for r in B: split_of.setdefault(r["value"], set()).add(r["split"])
    ok(all(len(s) == 1 for s in split_of.values()), "a value appears in two splits")
    ok(Counter(next(iter(s)) for s in split_of.values()) == Counter(SPLITS), f"split sizes {Counter(next(iter(s)) for s in split_of.values())}")
    ok(all(len(v) == 4 and v.isdigit() for v in split_of), "values not 4-digit")
    ok(set(kb) == {(c, v, i) for c in CONDS_B for v in split_of for i in range(N_ANS_B)}, "Arm B cells incomplete")

    # Ceiling: all T = 1 targets under none / ord, 80 finite log-probabilities <= 0 each
    C = jl(d / "armA_ceiling.jsonl"); kc = Counter((r["cond"], r["value"], r["idx"]) for r in C)
    tg = {(c, v, i) for (role, c, T, v, i) in exp if role == "target" and T == 1.0 and c in ("none", "ord")}
    ok(max(kc.values(), default=0) == 1, f"ceiling duplicates: {sum(v > 1 for v in kc.values())}")
    ok(set(kc) == tg, f"ceiling: {len(tg - set(kc))} targets missing")
    bad = [r for r in C if set(r["scores"]) != set(CANDS) or not all(math.isfinite(x) and x <= 1e-6 for x in r["scores"].values())]
    ok(not bad, f"ceiling rows with bad scores: {len(bad)}")

    # Decoders: every configuration predicts the full test set; probabilities are distributions
    test = {(v, i) for v, s in split_of.items() if "test" in s for i in range(N_ANS_B)}
    D = jl(d / "armB_decoder.jsonl"); groups = {}
    for r in D: groups.setdefault((r["decoder"], r["train_cond"], r["cond"], r["seed"], r["n_train"]), []).append((r["value"], r["idx"]))
    want = set()
    for c in CONDS_B:
        want |= {("direct", c, c, 0, n) for n in N_CURVE} | {("shuffled", c, c, 0, 500)} | {("direct", c, x, 0, 500) for x in CONDS_B if x != c}
        want |= {("direct", c, c, s, 500) for s in SEEDS_EXTRA.get(tag, [])}
    ok(set(groups) == want, f"Arm B decoder configs: missing {sorted(want - set(groups))}, unexpected {sorted(set(groups) - want)}")
    ok(all(len(g) == len(set(g)) and set(g) == test for g in groups.values()), "an Arm B decoder config does not cover the test set exactly once")
    AD = jl(d / "armA_decoder.jsonl"); ka = Counter((r["cond"], r["T"], r["value"], r["idx"]) for r in AD)
    ok(set(ka) == {(c, T, v, i) for c in CONDS_A for T in TEMPS_A for v in CANDS for i in range(N_TGT)} and max(ka.values()) == 1, "Arm A decoder rows incomplete")
    pr = [p for r in D for p in r["probs"]] + [p for r in AD for p in r["probs"]]
    ok(all(len(p) == 10 and abs(sum(p) - 1) < 1e-3 for p in pr), "decoder probabilities not distributions")
    ok(all(len(r["probs"]) == 4 for r in D) and all(len(r["probs"]) == 2 for r in AD), "decoder digit counts")

    # Descriptive watch rates
    print("truncated / empty: Arm A", f"{sum(r['truncated'] for r in A) / len(A):.2%} / {sum(not r['response'].strip() for r in A) / len(A):.2%}",
          "| Arm B", f"{sum(r['truncated'] for r in B) / len(B):.2%} / {sum(not r['response'].strip() for r in B) / len(B):.2%}")
    print("Arm B strata (1 disclosure, 2 exact, 3 partial, 4 none), share of answers:")
    for c in CONDS_B:
        s = Counter(stratum4(r["response"], r["value"]) for r in B if r["cond"] == c)
        print(f"  {c}: " + "  ".join(f"{k}: {s[k] / sum(s.values()):.1%}" for k in (1, 2, 3, 4)))
    print("Arm A strata (same coding, 2-digit), share of answers:")
    for c in CONDS_A:
        for T in TEMPS_A:
            s = Counter(stratum2(r["response"], r["value"], nums_clean(r["response"])) for r in A if r["cond"] == c and r["T"] == T)
            print(f"  {c} T={T}: " + "  ".join(f"{k}: {s[k] / sum(s.values()):.1%}" for k in (1, 2, 3, 4)))
    print("PASS: every integrity check" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return not fails

if __name__ == "__main__":
    sys.exit(0 if all([check(a) for a in sys.argv[1:]]) else 1)
