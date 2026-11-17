"""Family B through the graphical procedure under CORRELATED model outcomes (preregistration_v7.md §5, family B); design aid, no v7 data.

All five models use the same 48 pairs, so their pair-level differences d[m, p] can be correlated (e.g. a pair that triggers
copying in one model may trigger it in all). This checks, end to end (per-model sign-flip p -> graph_test on v7_stats.v7_graph, alpha_B = 0.025):
  - FWER of family B (any false rejection), and
  - the error of the "most models" claim (>= 3 rejections in the same direction when every model is null).
Two states of family P: not all rejected (B keeps 0.025) and all of P1-P3 rejected (B receives 0.05 in total).
Reported with false-rejection counts and Clopper-Pearson 95% Monte Carlo intervals.
Scenarios (all nulls, mean 0): sharp null, normal, cross-model correlation 0.5; copying-like contamination (5% of pairs, +5 nats,
offset to mean 0) independent across models; the same contamination on the SAME pairs in every model; shared centred exponential.
Pass criteria, fixed before the first run: "most models" error <= 0.05; FWER_B <= 0.035 with P not rejected and <= 0.07 with P
rejected (the nominal 0.025 / 0.05 with the same 1.4x tolerance as the other validations).
Usage: python analysis/v7_validate_familyB.py   -> results/v7_validate_familyB.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; sys.path.insert(0, str(ROOT / "analysis"))
import v7_stats as vs
M, N, N_REP, N_FLIP = 5, 48, 4000, 2000
CRIT = dict(most=0.05, fwer_p_not=0.035, fwer_p_rej=0.07)

def draw(sc, rng):
    if sc == "sharp null, normal, correlation 0.5":
        return np.sqrt(0.5) * rng.normal(0, 1.4, N)[None, :] + np.sqrt(0.5) * rng.normal(0, 1.4, (M, N))
    if sc == "copying-like, independent across models":
        return rng.normal(0, 1.0, (M, N)) + 5.0 * (rng.random((M, N)) < 0.05) - 0.25
    if sc == "copying-like, same pairs in every model":
        return rng.normal(0, 1.0, (M, N)) + (5.0 * (rng.random(N) < 0.05))[None, :] - 0.25
    if sc == "shared centred exponential (correlation 0.5)":
        return np.sqrt(0.5) * (rng.exponential(1.0, N) - 1)[None, :] + np.sqrt(0.5) * rng.normal(0, 1.0, (M, N))

def cp(k, n=N_REP):
    """Clopper-Pearson 95% interval for k false rejections in n replications (Monte Carlo uncertainty)."""
    from scipy import stats
    return [float(stats.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0, float(stats.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0]

def pvals(D, F):
    out = []
    for d in D:
        obs = abs(d.mean()); out.append(((np.abs((F * d).mean(1)) >= obs - 1e-12).sum() + 1) / (len(F) + 1))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(20261127); F = rng.choice([-1.0, 1.0], (N_FLIP, N)); B = [f"B{m}" for m in range(M)]; out = dict(criteria=CRIT, scenarios={})
    print(f"criteria: {CRIT}; {N_REP} replications, {M} models x {N} pairs")
    for sc in ["sharp null, normal, correlation 0.5", "copying-like, independent across models", "copying-like, same pairs in every model", "shared centred exponential (correlation 0.5)"]:
        k = {"P not rejected": [0, 0], "P rejected": [0, 0]}
        for _ in range(N_REP):
            D = draw(sc, rng); p = dict(zip(B, pvals(D, F))); sign = dict(zip(B, np.sign(D.mean(1))))
            for state, pp in [("P not rejected", 1.0), ("P rejected", 0.0)]:
                w, G = vs.v7_graph(B); rej = vs.graph_test({**p, "P1": pp, "P2": pp, "P3": pp}, w, G) - {"P1", "P2", "P3"}
                k[state][0] += bool(rej); k[state][1] += max(sum(1 for h in rej if sign[h] > 0), sum(1 for h in rej if sign[h] < 0)) >= 3
        r = {s: dict(fwer_B=v[0] / N_REP, fwer_B_count=v[0], fwer_B_ci95=cp(v[0]), most_models_error=v[1] / N_REP, most_models_count=v[1], most_models_ci95=cp(v[1]))
             for s, v in k.items()}
        r["pass"] = (r["P not rejected"]["fwer_B"] <= CRIT["fwer_p_not"] and r["P rejected"]["fwer_B"] <= CRIT["fwer_p_rej"]
                     and all(r[s]["most_models_error"] <= CRIT["most"] for s in k))
        out["scenarios"][sc] = r
        print(f"{'PASS' if r['pass'] else 'FAIL'} {sc}")
        for s in k:
            x = r[s]; print(f"   {s:15s} FWER_B {x['fwer_B_count']:4d}/{N_REP} = {x['fwer_B']:.4f} [{x['fwer_B_ci95'][0]:.4f}, {x['fwer_B_ci95'][1]:.4f}] | "
                            f"'most models' {x['most_models_count']:3d}/{N_REP} = {x['most_models_error']:.4f} [{x['most_models_ci95'][0]:.4f}, {x['most_models_ci95'][1]:.4f}]")
    out["all_pass"] = all(r["pass"] for r in out["scenarios"].values()); print("all pass:", out["all_pass"])
    (RES / "v7_validate_familyB.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
