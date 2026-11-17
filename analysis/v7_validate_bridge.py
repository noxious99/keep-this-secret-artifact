"""Validation of family B's per-model test (preregistration_v7.md §5, B1) before the freeze; design aid, no v7 data.

Unit: the pair. d_p = pair value under "secret" - under "reference number" (48 pairs, one model). The sign-flip test is exact under
the sharp null (both designations give the same answer distribution, so d_p is symmetric about 0). Under the weaker null of equal
MEANS with different distributions, d_p can be skewed and the test is only approximate. Checked here at the two levels that matter:
alpha = 0.05 and alpha = 0.005 (the initial per-model level of the graphical procedure, §5.0).

Tests compared: sign-flip of the mean (planned), studentized sign-flip, paired t; added after the first run showed all three
inflate under strong skew at 0.005: symmetric bootstrap-t (2,000 resamples) and Johnson (1978) skewness-corrected t.
Scenarios (mean 0 under the null): normal; t3; centered exponential; centered lognormal (sigma 1); pilot-like (difference of draws
from two DIFFERENT pilot residual pools, recentered); contaminated (5% of pairs +5 nats copying, recentered).
Pass criteria, fixed before the first run: FPR <= 0.07 at alpha 0.05 and <= 0.0075 at alpha 0.005.
Monte Carlo: 10,000 replications per scenario (SE about 0.0007 at 0.005); 4,000 flips per replication.
Usage: python analysis/v7_validate_bridge.py   -> results/v7_validate_bridge.json"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
spec = importlib.util.spec_from_file_location("pw", ROOT / "analysis" / "v7_power.py"); pw = importlib.util.module_from_spec(spec); spec.loader.exec_module(pw)
N, N_REP, N_FLIP = 48, 10000, 4000
CRIT = {0.05: 0.07, 0.005: 0.0075}

N_BOOT = 2000

def johnson_t(D):
    """Johnson (1978) skewness-corrected t for H0: mean 0, vectorized over rows; two-sided p from t(N-1)."""
    m = D.mean(1); s2 = D.var(1, ddof=1); mu3 = ((D - m[:, None]) ** 3).mean(1) * N / (N - 1)
    tj = (m + mu3 / (6 * s2 * N) + mu3 * m ** 2 / (3 * s2 ** 2)) / np.sqrt(s2 / N); return 2 * stats.t.sf(np.abs(tj), N - 1)

def tests(D, F, rng):
    """D: reps x N differences; F: flips x N signs. p values (reps x 5): sign-flip mean, studentized sign-flip, paired t,
    symmetric bootstrap-t (|t*| of resamples centred at the sample mean), Johnson's t."""
    m = D.mean(1); s = D.std(1, ddof=1); t = m / (s / np.sqrt(N)); out = np.empty((len(D), 5))
    for i in range(len(D)):
        X = F * D[i]; mf = X.mean(1); sf = X.std(1, ddof=1); tf = mf / (sf / np.sqrt(N))
        out[i, 0] = ((np.abs(mf) >= abs(m[i]) - 1e-12).sum() + 1) / (len(F) + 1)
        out[i, 1] = ((np.abs(tf) >= abs(t[i]) - 1e-12).sum() + 1) / (len(F) + 1)
        B = D[i][rng.integers(0, N, (N_BOOT, N))]; tb = (B.mean(1) - m[i]) / (B.std(1, ddof=1) / np.sqrt(N))
        out[i, 3] = ((np.abs(tb) >= abs(t[i])).sum() + 1) / (N_BOOT + 1)
    out[:, 2] = 2 * stats.t.sf(np.abs(t), N - 1); out[:, 4] = johnson_t(D); return out

def scenario(name, rng, inp):
    if name == "normal": D = rng.normal(0, 1.4, (N_REP, N))
    elif name == "t3": D = rng.standard_t(3, (N_REP, N))
    elif name == "centered exponential": D = rng.exponential(1.0, (N_REP, N)) - 1.0
    elif name == "centered lognormal (sigma 1)": D = rng.lognormal(0, 1, (N_REP, N)) - np.exp(0.5)
    elif name == "pilot-like (two different residual pools)":
        a, b = inp["resid"]["phi-4"] * 1.5, inp["resid"]["llama-31-8b-instruct"]
        D = rng.choice(a, (N_REP, N)) - rng.choice(b, (N_REP, N)) - (a.mean() - b.mean())
    elif name == "contaminated (5% of pairs +5 nats)":
        D = rng.normal(0, 1.0, (N_REP, N)) + 5.0 * (rng.random((N_REP, N)) < 0.05) - 0.25
    return D

if __name__ == "__main__":
    rng = np.random.default_rng(20261126); inp = pw.pilot_inputs(); F = rng.choice([-1.0, 1.0], (N_FLIP, N)); out = dict(criteria={str(k): v for k, v in CRIT.items()}, scenarios={})
    names = ["sign-flip (mean)", "studentized sign-flip", "paired t", "bootstrap-t (symmetric)", "Johnson t"]
    print(f"criteria: FPR <= {CRIT[0.05]} at 0.05 and <= {CRIT[0.005]} at 0.005; {N_REP} replications, {N} pairs")
    for sc in ["normal", "t3", "centered exponential", "centered lognormal (sigma 1)", "pilot-like (two different residual pools)", "contaminated (5% of pairs +5 nats)"]:
        D = scenario(sc, rng, inp); P = tests(D, F, rng); r = {}
        for j, nm in enumerate(names):
            f05, f005 = float((P[:, j] <= 0.05).mean()), float((P[:, j] <= 0.005).mean())
            r[nm] = dict(fpr_05=f05, fpr_005=f005, passed=f05 <= CRIT[0.05] and f005 <= CRIT[0.005])
        out["scenarios"][sc] = dict(skewness=float(stats.skew(D.ravel())), tests=r)
        print(f"== {sc} (skewness of d {out['scenarios'][sc]['skewness']:+.2f})")
        for nm, v in r.items(): print(f"   {'PASS' if v['passed'] else 'FAIL'} {nm:24s} FPR at 0.05 {v['fpr_05']:.4f} | at 0.005 {v['fpr_005']:.4f}")
    out["all_pass"] = {nm: all(s["tests"][nm]["passed"] for s in out["scenarios"].values()) for nm in names}
    print("all pass:", out["all_pass"])
    (RES / "v7_validate_bridge.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
