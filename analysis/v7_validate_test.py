"""Validation of v7's primary test and intervals (preregistration_v7.md §4.2) before the freeze; design aid, no v7 data.

The label-level exact permutation test (16 pooled label means, C(16,8) assignments) and the t intervals (df 14) are checked under:
shared label effects, heavy tails, heterogeneous models, class-dependent variance and pair x class interaction (the one term the
label-level analysis does not model), separately and combined.

Pass criteria, fixed before the first run:
  - false-positive rate (true difference 0, alpha 0.05) <= 0.07;
  - 95% CI coverage (true difference 0 and 0.5 nats) within 0.92-0.98;
  - false equivalence (true difference 0.5 = the margin; 90% CI inside +-0.5) <= 0.07 (TOST error at the boundary).
A scenario that fails is reported, not tuned away; preregistration_v7.md is then changed before the freeze.
First run (label-level only): fails under a pair x class interaction shared across models (FPR 0.12 at the pilot-sized 0.5, 0.20 at 1.0),
and over-covers (conservative) with one outlier label. Second analysis added after that run, before any v7 data: the CROSSED version,
  Z[l, p] = label x pair value pooled over models; d_p = mean_sens Z[:, p] - mean_sal Z[:, p];
  sigma2_res = within-class two-way ANOVA residual of Z (mean of the two classes); sigma2_g = max(0, var_p(d_p) - sigma2_res * 2/8);
  SE^2 = s2_w * 2/8 + sigma2_g / P; t test and t intervals with df 14 (the label df; conservative).
The same criteria apply to it.
Monte Carlo: 2,000 replications per scenario (SE about 0.005 at 0.05).
Usage: python analysis/v7_validate_test.py   -> results/v7_validate_test.json"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
spec = importlib.util.spec_from_file_location("pw", ROOT / "analysis" / "v7_power.py"); pw = importlib.util.module_from_spec(spec); spec.loader.exec_module(pw)
N_SIM, N_MODELS, N_LAB, P = 2000, 5, 8, 24
CRIT = dict(fpr_max=0.07, cov_lo=0.92, cov_hi=0.98, false_equiv_max=0.07)

def t_scaled(rng, df, size, var):
    """Student-t draws scaled to the given variance (heavy tails, same variance)."""
    return rng.standard_t(df, size) * np.sqrt(var * (df - 2) / df)

def draw(sc, inp, delta, rng):
    """One simulated v7: the 5 x 16 matrix of label means (first 8 sensitive)."""
    s2 = sc.get("label_var", inp["sig2_label"]); rho = sc.get("rho", inp["rho"])
    cls_scale = np.r_[np.ones(N_LAB), np.full(N_LAB, np.sqrt(sc.get("salient_var_ratio", 1.0)))]
    if sc.get("label_tails"): a_sh = t_scaled(rng, 3, 2 * N_LAB, rho * s2)
    else: a_sh = rng.normal(0, np.sqrt(rho * s2), 2 * N_LAB)
    if sc.get("outlier_label"):                                   # one label (either class, at random) leaks +4 nats more in every model
        a_sh[rng.integers(2 * N_LAB)] += 4.0
    eff = np.r_[np.full(N_LAB, delta / 2), np.full(N_LAB, -delta / 2)]
    res_scale = sc.get("model_resid_scale", [1.0] * N_MODELS); lab_m = sc.get("model_label_var", [(1 - rho) * s2] * N_MODELS)
    g_shared = rng.normal(0, np.sqrt(sc.get("pairclass_shared", 0.0)), P)
    Z = np.empty((N_MODELS, 2 * N_LAB, P))
    for m in range(N_MODELS):
        pool = inp["resid"][pw.MODELS_PILOT[m % 2]] * res_scale[m]
        E = rng.choice(pool, (2 * N_LAB, P))
        g = g_shared + rng.normal(0, np.sqrt(sc.get("pairclass_model", 0.0)), P)      # pair x class interaction on the class difference
        E += np.r_[np.full(N_LAB, 0.5), np.full(N_LAB, -0.5)][:, None] * g[None, :]
        Z[m] = (eff + cls_scale * (a_sh + rng.normal(0, np.sqrt(lab_m[m]), 2 * N_LAB)))[:, None] + E
    return Z.mean(0)                                              # 16 x P: label x pair values pooled over models

def crossed_ci(Zp, level):
    """Crossed interval for sens - sal from the pooled 16 x P table (label and pair x class sampling); returns lo, hi, p."""
    yb = Zp.mean(1); a, b = yb[:N_LAB], yb[N_LAB:]; s2w = (a.var(ddof=1) + b.var(ddof=1)) / 2; d = a.mean() - b.mean()
    res = []
    for blk in (Zp[:N_LAB], Zp[N_LAB:]):
        r = blk - blk.mean(1, keepdims=True) - blk.mean(0, keepdims=True) + blk.mean(); res.append((r ** 2).sum() / ((N_LAB - 1) * (Zp.shape[1] - 1)))
    dp = Zp[:N_LAB].mean(0) - Zp[N_LAB:].mean(0); s2g = max(0.0, dp.var(ddof=1) - np.mean(res) * 2 / N_LAB)
    se = np.sqrt(s2w * 2 / N_LAB + s2g / Zp.shape[1]); df = 2 * N_LAB - 2; h = stats.t.ppf(0.5 + level / 2, df) * se
    return d - h, d + h, float(2 * stats.t.sf(abs(d) / se, df))

def run(sc, inp, rng):
    out = {}
    for delta in (0.0, 0.5):
        rej = rej_rank = cov95 = cov90 = equiv = 0; xr = xc95 = xc90 = xeq = 0
        for _ in range(N_SIM):
            Zp = draw(sc, inp, delta, rng); yb = Zp.mean(1)
            lo, hi, px = crossed_ci(Zp, 0.95); xr += px < 0.05; xc95 += lo <= delta <= hi
            lo, hi, _ = crossed_ci(Zp, 0.90); xc90 += lo <= delta <= hi; xeq += (-0.5 < lo) and (hi < 0.5)
            p = pw.perm_p(yb); rej += p < 0.05; rej_rank += pw.perm_p(yb, rank=True) < 0.05
            lo, hi, _ = pw.label_ci(yb, 0.95); cov95 += lo <= delta <= hi
            lo9, hi9, _ = pw.label_ci(yb, 0.90); cov90 += lo9 <= delta <= hi9; equiv += (-0.5 < lo9) and (hi9 < 0.5)
        out[f"d{delta}"] = dict(reject=rej / N_SIM, reject_rank=rej_rank / N_SIM, cov95=cov95 / N_SIM, cov90=cov90 / N_SIM, equivalent=equiv / N_SIM,
                                 crossed=dict(reject=xr / N_SIM, cov95=xc95 / N_SIM, cov90=xc90 / N_SIM, equivalent=xeq / N_SIM))
    r0, r5 = out["d0.0"], out["d0.5"]
    out["checks"] = dict(fpr=r0["reject"] <= CRIT["fpr_max"], cov95_null=CRIT["cov_lo"] <= r0["cov95"] <= CRIT["cov_hi"],
                         cov95_alt=CRIT["cov_lo"] <= r5["cov95"] <= CRIT["cov_hi"], false_equiv=r5["equivalent"] <= CRIT["false_equiv_max"])
    out["pass"] = all(out["checks"].values()); x0, x5 = r0["crossed"], r5["crossed"]
    out["checks_crossed"] = dict(fpr=x0["reject"] <= CRIT["fpr_max"], cov95_null=CRIT["cov_lo"] <= x0["cov95"] <= CRIT["cov_hi"],
                                 cov95_alt=CRIT["cov_lo"] <= x5["cov95"] <= CRIT["cov_hi"], false_equiv=x5["equivalent"] <= CRIT["false_equiv_max"])
    out["pass_crossed"] = all(out["checks_crossed"].values()); return out

def greedy_run(sc, inp, rng):
    """Greedy identification: binomial over 40 candidates, model-specific base rates, label effects shared/model-specific."""
    p0 = np.array(sc["p0"]); lv = sc.get("label_var", inp["greedy_label"]); out = {}
    for delta in (0.0, 0.20):
        rej = cov = 0
        for _ in range(N_SIM):
            a = rng.normal(0, np.sqrt(lv), 2 * N_LAB); eff = np.r_[np.full(N_LAB, delta / 2), np.full(N_LAB, -delta / 2)]
            Y = np.stack([rng.binomial(40, np.clip(p0[m] + eff + a + rng.normal(0, np.sqrt(lv / 2), 2 * N_LAB), 0.01, 0.99)) / 40 for m in range(N_MODELS)])
            yb = Y.mean(0); rej += pw.perm_p(yb) < 0.05; lo, hi, _ = pw.label_ci(yb, 0.95); cov += lo <= delta <= hi
        out[f"d{delta}"] = dict(reject=rej / N_SIM, cov95=cov / N_SIM)
    out["pass"] = out["d0.0"]["reject"] <= CRIT["fpr_max"] and all(CRIT["cov_lo"] <= out[k]["cov95"] <= CRIT["cov_hi"] for k in ("d0.0", "d0.2"))
    return out

def greedy_crossed_run(sc, inp, rng, n_cand=40):
    """Greedy, crossed form (preregistration_v7.md §4.2 with candidates as the crossed unit): per model a 16 x 40 table of 0/1 replay
    scores, p = base rate (model) + class effect + label effect (shared + model-specific) + candidate main effect (shared) + candidate x
    class interaction (shared and/or model-specific). Pilot (2 models): candidate main-effect SD 0.13-0.20; interaction variance
    0.003-0.053; cross-model correlation of candidate-level class differences -0.02 / +0.19."""
    p0 = np.array(sc["p0"]); lv = sc.get("label_var", inp["greedy_label"]); out = {}
    for delta in (0.0, 0.20):
        k = dict(rej=0, rej_l=0, eq=0); iv, iv_l, truth = [], [], []      # coverage against the true difference of CLIPPED probabilities
        for _ in range(N_SIM):
            eff = np.r_[np.full(N_LAB, delta / 2), np.full(N_LAB, -delta / 2)]; a = rng.normal(0, np.sqrt(lv / 2), 2 * N_LAB)
            c = rng.normal(0, sc.get("cand_sd", 0.0), n_cand); g = rng.normal(0, np.sqrt(sc.get("candclass_shared", 0.0)), n_cand)
            sign = np.r_[np.full(N_LAB, 0.5), np.full(N_LAB, -0.5)]; Z = np.zeros((2 * N_LAB, n_cand))
            for m in range(N_MODELS):
                gm = g + rng.normal(0, np.sqrt(sc.get("candclass_model", 0.0)), n_cand)
                pr = p0[m] + (eff + a + rng.normal(0, np.sqrt(lv / 2), 2 * N_LAB))[:, None] + c[None, :] + sign[:, None] * gm[None, :]
                pc = np.clip(pr, 0.01, 0.99); Z += rng.random(pr.shape) < pc; P_true = pc if m == 0 else P_true + pc
            Zp = Z / N_MODELS; P_true = P_true / N_MODELS; truth.append(P_true[:N_LAB].mean() - P_true[N_LAB:].mean())
            lo, hi, px = crossed_ci(Zp, 0.95); k["rej"] += px < 0.05; iv.append((lo, hi))
            lo9, hi9, _ = crossed_ci(Zp, 0.90); k["eq"] += (-0.10 < lo9) and (hi9 < 0.10)
            yb = Zp.mean(1); k["rej_l"] += pw.perm_p(yb) < 0.05; iv_l.append(pw.label_ci(yb, 0.95)[:2])
        t = float(np.mean(truth)) if delta else 0.0
        out[f"d{delta}"] = {key: v / N_SIM for key, v in k.items()}
        out[f"d{delta}"].update(true_delta=t, cov=float(np.mean([lo <= t <= hi for lo, hi in iv])), cov_l=float(np.mean([lo <= t <= hi for lo, hi in iv_l])))
    a0, a2 = out["d0.0"], out["d0.2"]
    out["pass_crossed"] = a0["rej"] <= CRIT["fpr_max"] and all(CRIT["cov_lo"] <= x["cov"] for x in (a0, a2)) and all(x["cov"] <= CRIT["cov_hi"] for x in (a0, a2))
    out["valid_or_conservative"] = a0["rej"] <= CRIT["fpr_max"] and all(CRIT["cov_lo"] <= x["cov"] for x in (a0, a2))
    out["pass_label_level"] = a0["rej_l"] <= CRIT["fpr_max"] and all(CRIT["cov_lo"] <= x["cov_l"] <= CRIT["cov_hi"] for x in (a0, a2))
    return out

SCENARIOS = {
    "A baseline (pilot inputs)": {},
    "B label effects fully shared (rho 1)": dict(rho=1.0),
    "C label effects model-specific only (rho 0)": dict(rho=0.0),
    "D heavy-tailed label effects (t3)": dict(label_tails=True),
    "E one outlier label, +4 nats in every model": dict(outlier_label=True),
    "F heavier residuals (x2 in every model)": dict(model_resid_scale=[2.0] * 5),
    "G heterogeneous models (residual x0.5..x3; one model label var 2.24)": dict(model_resid_scale=[0.5, 1.0, 1.0, 2.0, 3.0], model_label_var=[0.05, 0.05, 0.05, 0.05, 2.24]),
    "H salient labels 4x the label variance of sensitive": dict(salient_var_ratio=4.0),
    "I pair x class interaction 0.5, model-specific (pilot max)": dict(pairclass_model=0.5),
    "J pair x class interaction 0.5, shared across models": dict(pairclass_shared=0.5),
    "K pair x class interaction 1.0, shared across models": dict(pairclass_shared=1.0),
    "L combined: D + G + H + I": dict(label_tails=True, model_resid_scale=[0.5, 1.0, 1.0, 2.0, 3.0], model_label_var=[0.05, 0.05, 0.05, 0.05, 2.24], salient_var_ratio=4.0, pairclass_model=0.5),
}
GREEDY = {
    "g1 baseline (base rate 0.62 in every model)": dict(p0=[0.62] * 5),
    "g2 heterogeneous base rates (0.45-0.85)": dict(p0=[0.45, 0.55, 0.65, 0.75, 0.85]),
    "g3 heterogeneous + label var 0.05": dict(p0=[0.45, 0.55, 0.65, 0.75, 0.85], label_var=0.05),
}

GREEDY_CROSSED = {
    "gc1 baseline (base 0.62, candidate SD 0.2)": dict(p0=[0.62] * 5, cand_sd=0.2),
    "gc2 heterogeneous base rates 0.45-0.85": dict(p0=[0.45, 0.55, 0.65, 0.75, 0.85], cand_sd=0.2),
    "gc3 interaction 0.05 model-specific (pilot max)": dict(p0=[0.62] * 5, cand_sd=0.2, candclass_model=0.05),
    "gc4 interaction 0.01 shared across models": dict(p0=[0.62] * 5, cand_sd=0.2, candclass_shared=0.01),
    "gc5 interaction 0.05 shared across models (pilot max, all shared)": dict(p0=[0.62] * 5, cand_sd=0.2, candclass_shared=0.05),
    "gc6 combined: heterogeneous + label var 0.05 + gc5": dict(p0=[0.45, 0.55, 0.65, 0.75, 0.85], cand_sd=0.2, label_var=0.05, candclass_shared=0.05),
}

if __name__ == "__main__":
    rng = np.random.default_rng(20261124); inp = pw.pilot_inputs(); out = dict(criteria=CRIT, n_sim=N_SIM, kl={}, greedy={})
    print(f"criteria: {CRIT}; {N_SIM} replications per scenario")
    for name, sc in SCENARIOS.items():
        r = run(sc, inp, rng); out["kl"][name] = r; a, b = r["d0.0"], r["d0.5"]
        print(f"{'PASS' if r['pass'] else 'FAIL'} {name}\n     null: FPR {a['reject']:.3f} (rank {a['reject_rank']:.3f}), 95% cov {a['cov95']:.3f}, 90% cov {a['cov90']:.3f} | "
              f"delta 0.5: power {b['reject']:.2f}, 95% cov {b['cov95']:.3f}, false equivalence {b['equivalent']:.3f}" + ("" if r["pass"] else f"  failed: {[k for k, v in r['checks'].items() if not v]}"))
        xa, xb = a["crossed"], b["crossed"]
        print(f"  {'PASS' if r['pass_crossed'] else 'FAIL'} crossed: null FPR {xa['reject']:.3f}, 95% cov {xa['cov95']:.3f}, 90% cov {xa['cov90']:.3f} | "
              f"delta 0.5: power {xb['reject']:.2f}, 95% cov {xb['cov95']:.3f}, false equivalence {xb['equivalent']:.3f}" + ("" if r["pass_crossed"] else f"  failed: {[k for k, v in r['checks_crossed'].items() if not v]}"))
    for name, sc in GREEDY.items():
        r = greedy_run(sc, inp, rng); out["greedy"][name] = r
        print(f"{'PASS' if r['pass'] else 'FAIL'} {name}: null FPR {r['d0.0']['reject']:.3f}, 95% cov {r['d0.0']['cov95']:.3f} | delta 20 pp: power {r['d0.2']['reject']:.2f}, 95% cov {r['d0.2']['cov95']:.3f}")
    out["greedy_crossed"] = {}
    for name, sc in GREEDY_CROSSED.items():
        r = greedy_crossed_run(sc, inp, rng); out["greedy_crossed"][name] = r; a0, a2 = r["d0.0"], r["d0.2"]
        print(f"{'PASS' if r['pass_crossed'] else ('CONS' if r['valid_or_conservative'] else 'FAIL')} {name}: crossed null FPR {a0['rej']:.3f}, 95% cov {a0['cov']:.3f}/{a2['cov']:.3f}, "
              f"power at 20 pp (true {100 * a2['true_delta']:.1f}) {a2['rej']:.2f}, equivalent at +-10 pp (null) {a0['eq']:.3f} | label-level FPR {a0['rej_l']:.3f}, cov {a0['cov_l']:.3f}")
    out["greedy_crossed_valid_or_conservative"] = all(r["valid_or_conservative"] for r in out["greedy_crossed"].values())
    out["all_pass_label_level"] = all(r["pass"] for r in out["kl"].values())
    out["all_pass_crossed"] = all(r["pass_crossed"] for r in out["kl"].values()); out["all_pass_greedy"] = all(r["pass"] for r in out["greedy"].values())
    # a failure by over-coverage alone (coverage > 0.98, FPR and false equivalence within limits) is conservative, not invalid
    out["crossed_valid_or_conservative"] = all(x["checks_crossed"]["fpr"] and x["checks_crossed"]["false_equiv"] and x["d0.0"]["crossed"]["cov95"] >= CRIT["cov_lo"]
                                               and x["d0.5"]["crossed"]["cov95"] >= CRIT["cov_lo"] for x in out["kl"].values())
    print({k: v for k, v in out.items() if k.startswith("all_pass") or k.startswith("crossed_valid") or k.startswith("greedy_crossed_valid")})
    (RES / "v7_validate_test.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
