"""EXPLORATORY follow-ups to the pre-registered attacker-study analysis (written 2026-11-09, after results/attack_analysis.json was seen).
Nothing here is confirmatory. Reuses the frozen helpers of analysis/attack_analyze.py unchanged.

  E1  the majority baseline: its digits, hits on the test values, accuracy vs the 0.10 chance level
  E2  Arm B decoder vs its own shuffled-label control and vs 0.10 (paired / one-sample sign-flip over test values; Holm over models),
      on strata 2-4 and on stratum 4 (literal-free); for ref, stratum 4 vs majority is the literal-free analogue of H-A2
  E3  ref vs ord, paired over the 200 test values (the same values in both conditions); decoder and frequency-Bayes
  E4  decoder seeds (Llama): accuracy per seed vs majority and vs 0.10
  E5  per-position decoder accuracy
  E6  answer-level decoder accuracy by occurrence stratum (2 exact vs 4 none)
  E7  calibration: mean top-digit probability vs accuracy (answer level)
  E8  Arm A required-secondary above-chance tests with Holm over the 5 models (attack_analyze.py reports unadjusted p only)
  E9  instruction effect (ord - none) at the attacker level vs v5/v6 KL (P1, clean): ceiling at T = 1 and greedy replay
  E10 digit avoidance: share of parsed 4-digit numbers whose digit matches the value's digit at the same position (chance 0.10),
      and the cross-condition decoder (trained on the other condition) vs 0.10, two-sided (added after a transfer below chance was seen)

Usage:
  python analysis/attack_explore.py              -> results/attack_explore.json (needs results/attack_analysis.json)
  python analysis/attack_explore.py --selftest   synthetic null / planted data (attack_analyze.make_fake)"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_analyze import jl, stratum4, nums4, value_metrics, paired, logpost_freq, freq_logprobs, make_fake, MODELS, ALPHA
from r0_common import RES, holm, jsonable
sys.stdout.reconfigure(encoding="utf-8")
CHANCE = 0.10

def load_b(d, cond):
    rows = [r for r in jl(d / "armB_sampled.jsonl") if r["cond"] == cond]
    train = [r for r in rows if r["split"] == "train"]; test = sorted([r for r in rows if r["split"] == "test"], key=lambda r: (r["value"], r["idx"]))
    for r in test: r["_s"] = stratum4(r["response"], r["value"])
    pm = {}
    for p in jl(d / "armB_decoder.jsonl"):
        if p["cond"] == cond and p["train_cond"] == cond: pm[(p["decoder"], p["seed"], p["n_train"], p["value"], p["idx"])] = np.array(p["probs"])
    return train, test, pm

def majority(train, test):
    tv = sorted({r["value"] for r in test}); maj = [int(pd.Series([int(r["value"][j]) for r in train]).value_counts().idxmax()) for j in range(4)]
    hits = [sum(int(v[j]) == maj[j] for v in tv) for j in range(4)]
    return dict(digits=maj, hits_per_position=hits, n_values=len(tv), accuracy=sum(hits) / (4 * len(tv)), per_value={v: float(np.mean([int(v[j]) == maj[j] for j in range(4)])) for v in tv})

def per_value_acc(test, pm, keep, kind="direct", seed=0):
    tv = sorted({r["value"] for r in test}); truth = {v: np.array([int(c) for c in v]) for v in tv}; out = {v: [] for v in tv}
    for r in test:
        if keep(r): out[r["value"]].append(np.log(np.clip(pm[(kind, seed, 500, r["value"], r["idx"])], 1e-12, 1)))
    return {v: m["acc"] for v, m in value_metrics(out, truth).items()}

S24 = lambda r: r["_s"] in (2, 3, 4); S4 = lambda r: r["_s"] == 4

def explore_b(d):
    R = {}
    for cond in ("ord", "ref"):
        train, test, pm = load_b(d, cond); mj = majority(train, test); c = dict(majority=dict((k, v) for k, v in mj.items() if k != "per_value"))
        for label, keep in (("strata_2_4", S24), ("stratum_4", S4)):
            dm, sm = per_value_acc(test, pm, keep), per_value_acc(test, pm, keep, "shuffled")
            c[label] = dict(decoder=float(np.mean(list(dm.values()))), shuffled=float(np.mean(list(sm.values()))), n_values=len(dm),
                            vs_shuffled=paired([dm[v] - sm[v] for v in dm if v in sm]), vs_chance=paired([dm[v] - CHANCE for v in dm]),
                            vs_majority=paired([dm[v] - mj["per_value"][v] for v in dm]))
        seeds = sorted({k[1] for k in pm if k[0] == "direct" and k[2] == 500})
        if len(seeds) > 1:
            c["seeds"] = {}
            for s in seeds:
                a = per_value_acc(test, pm, S24, seed=s)
                c["seeds"][str(s)] = dict(acc=float(np.mean(list(a.values()))), vs_majority=float(np.mean([a[v] - mj["per_value"][v] for v in a])), vs_chance=float(np.mean(list(a.values()))) - CHANCE)
        pos = np.zeros(4); n = 0; by_s = {2: [], 4: []}; conf, hit = [], []
        for r in test:
            if not S24(r): continue
            p = pm[("direct", 0, 500, r["value"], r["idx"])]; pred = p.argmax(1); tr = np.array([int(ch) for ch in r["value"]])
            pos += pred == tr; n += 1; a = float(np.mean(pred == tr)); conf.append(float(p.max(1).mean())); hit.append(a)
            if r["_s"] in by_s: by_s[r["_s"]].append(a)
        c["per_position"] = (pos / max(n, 1)).tolist()
        c["answer_level_by_stratum"] = {str(s): dict(n_answers=len(v), acc=float(np.mean(v)) if v else float("nan")) for s, v in by_s.items()}
        c["calibration"] = dict(mean_top_prob=float(np.mean(conf)), mean_acc=float(np.mean(hit)), n_answers=n)
        match = [float(np.mean([s[j] == r["value"][j] for j in range(4)])) for r in test if r["_s"] != 1 for s in nums4(r["response"])]
        c["digit_match_rate"] = dict(rate=float(np.mean(match)) if match else float("nan"), n_numbers=len(match))
        other = "ref" if cond == "ord" else "ord"; xm = {}
        for p in jl(d / "armB_decoder.jsonl"):
            if p["cond"] == cond and p["train_cond"] == other and p["decoder"] == "direct": xm[(p["value"], p["idx"])] = np.array(p["probs"])
        if xm:
            xa = {}
            for r in test:
                if S24(r): xa.setdefault(r["value"], []).append(float(np.mean(xm[(r["value"], r["idx"])].argmax(1) == np.array([int(ch) for ch in r["value"]]))))
            c["transfer_vs_chance"] = dict(paired([np.mean(v) - CHANCE for v in xa.values()], one_sided=False), train_cond=other)
        R[cond] = dict(c, _dm24=per_value_acc(test, pm, S24), _dm4=per_value_acc(test, pm, S4),
                       _fb=freq_value_acc(train, test))
    # E3: ref vs ord, paired over test values (two-sided)
    a, b = R["ord"], R["ref"]
    R["ref_minus_ord"] = {lab: paired([b[k][v] - a[k][v] for v in a[k] if v in b[k]], one_sided=False) for lab, k in (("decoder_strata_2_4", "_dm24"), ("decoder_stratum_4", "_dm4"), ("freq_bayes_strata_2_4", "_fb"))}
    for cond in ("ord", "ref"):
        for k in ("_dm24", "_dm4", "_fb"): R[cond].pop(k)
    return R

def freq_value_acc(train, test):
    LP = logpost_freq(train); tv = sorted({r["value"] for r in test}); truth = {v: np.array([int(c) for c in v]) for v in tv}; out = {v: [] for v in tv}
    for r in test:
        if S24(r): out[r["value"]].append(freq_logprobs(LP, r["response"]))
    return {v: m["acc"] for v, m in value_metrics(out, truth).items()}

def holm_family(per_model, path):
    names = [m for m in per_model if get(per_model[m], path) is not None]; ps = [get(per_model[m], path)["p"] for m in names]
    return {m: dict(get(per_model[m], path), p_holm=p) for m, p in zip(names, holm(ps) if ps else [])}
def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d

def run(dirs, analysis=None, r0=None):
    per = {name: explore_b(Path(d)) for name, d in dirs.items()}
    fam = {}
    for cond in ("ord", "ref"):
        for lab in ("strata_2_4", "stratum_4"):
            for cmp in ("vs_shuffled", "vs_chance", "vs_majority"):
                fam[f"{cond}|{lab}|{cmp}"] = holm_family(per, (cond, lab, cmp))
    for lab in ("decoder_strata_2_4", "decoder_stratum_4", "freq_bayes_strata_2_4"): fam[f"ref_minus_ord|{lab}"] = holm_family(per, ("ref_minus_ord", lab))
    out = dict(per_model=per, families=fam)
    if analysis:                                                   # E8, E9
        A = {m: analysis["per_model"][m]["arm_a"] for m in per if m in analysis["per_model"]}; e8 = {}
        for cond in ("none", "ord", "ref"):
            for T in ("1.0", "0.3"):
                key = f"{cond}|T{T}"; ms = [m for m in A if key in A[m]]; ps = [A[m][key]["primary_freq"]["p_perm"] for m in ms]
                e8[key] = {m: dict(acc=A[m][key]["primary_freq"]["acc"], p=p, p_holm=ph) for m, p, ph in zip(ms, ps, holm(ps))}
        out["arm_a_above_chance_holm"] = e8
        if r0:
            e9 = {}
            for m in A:
                if m not in r0["estimands"]: continue
                dkl = r0["estimands"][m]["contrasts"]["P1_ord-none"]["clean"]["estimate"]; cn, co = A[m].get("none|T1.0", {}).get("ceiling"), A[m].get("ord|T1.0", {}).get("ceiling")
                g = A[m]["greedy_replay"]; dc = co - cn if cn is not None and co is not None else None; dr = g["ord"] - g["none"]
                e9[m] = dict(dKL_P1_clean=dkl, ceiling_none=cn, ceiling_ord=co, d_ceiling_pp=dc, replay_none=g["none"], replay_ord=g["ord"], d_replay_pp=dr,
                             ceiling_sign_matches_KL=None if dc is None else bool(np.sign(dc) == np.sign(dkl)), replay_sign_matches_KL=bool(np.sign(dr) == np.sign(dkl)))
            out["instruction_effect_vs_KL"] = e9
    return out

def selftest():
    ok = []
    def check(name, cond, info=""): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, info)
    with tempfile.TemporaryDirectory() as tmp:
        res = {}
        for mode in ("null", "planted"):
            dirs = {f"{mode}{i}": make_fake(Path(tmp) / f"{mode}{i}", mode, 100 + i)[0] for i in range(5)}
            res[mode] = run(dirs); print(mode, "done")
        F = {m: res[m]["families"] for m in res}
        sig = lambda f: sum(v["p_holm"] < ALPHA for v in f.values())
        n_null = sum(sig(F["null"][k]) for k in F["null"] if k.endswith(("vs_shuffled", "vs_chance")))
        check("null: decoder vs shuffled / vs 0.10, Holm-significant model-cells <= 1 of 40", n_null <= 1, f"({n_null})")
        print("  info: null decoder vs majority, Holm-significant model-cells:", sum(sig(F["null"][k]) for k in F["null"] if k.endswith("vs_majority")), "of 20")
        check("null: ref - ord not significant", all(sig(F["null"][k]) <= 1 for k in F["null"] if k.startswith("ref_minus_ord")))
        for cond in ("ord", "ref"):
            check(f"planted: {cond} decoder > shuffled in 5/5 (Holm)", sig(F["planted"][f"{cond}|strata_2_4|vs_shuffled"]) == 5)
            check(f"planted: {cond} decoder > 0.10 in 5/5 (Holm)", sig(F["planted"][f"{cond}|strata_2_4|vs_chance"]) == 5)
        check("planted: ref leaks less than ord (ref - ord < 0 in 5/5)", all(v["mean"] < 0 for v in F["planted"]["ref_minus_ord|decoder_strata_2_4"].values()))
        mb = [res["null"]["per_model"][m]["ord"]["majority"]["accuracy"] for m in res["null"]["per_model"]]
        check("null: majority baseline near 0.10 on average", abs(np.mean(mb) - CHANCE) < 0.02, f"({np.round(mb, 3).tolist()})")
        dmn = [res["null"]["per_model"][m]["ord"]["digit_match_rate"]["rate"] for m in res["null"]["per_model"]]
        dmp = [res["planted"]["per_model"][m]["ord"]["digit_match_rate"]["rate"] for m in res["planted"]["per_model"]]
        check("E10: digit-match rate about 0.10 in null, above 0.15 in planted (ord leak 0.10)", all(abs(x - CHANCE) < 0.005 for x in dmn) and all(x > 0.15 for x in dmp),
              f"(null {np.round(dmn, 4).tolist()}, planted {np.round(dmp, 3).tolist()})")
        cal = [res["planted"]["per_model"][m]["ord"]["calibration"] for m in res["planted"]["per_model"]]
        check("calibration fields finite", all(np.isfinite([c["mean_top_prob"], c["mean_acc"]]).all() for c in cal))
    print(f"{sum(ok)}/{len(ok)} checks passed"); return all(ok)

if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    dirs = {name: RES / f"attack_{tag}" for name, (tag, _) in MODELS.items() if (RES / f"attack_{tag}").exists()}
    analysis = json.load(open(RES / "attack_analysis.json", encoding="utf-8")); r0 = json.load(open(RES / "r0_audit.json", encoding="utf-8"))
    out = run(dirs, analysis, r0)
    out["meta"] = dict(script="analysis/attack_explore.py", status="EXPLORATORY: written after results/attack_analysis.json was seen", chance=CHANCE)
    (RES / "attack_explore.json").write_text(json.dumps(jsonable(out), indent=1), encoding="utf-8")
    for k, f in out["families"].items(): print(k, {m: (round(v["mean"], 4), round(v["p_holm"], 4)) for m, v in f.items()})
    print("wrote", RES / "attack_explore.json")
