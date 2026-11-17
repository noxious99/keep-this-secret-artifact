"""v7 power simulation from the label-variance pilot (analysis/v7_pilot_plan.md §3; results/v7pilot_analysis.json). Design aid.

Model of a label's mean LLR in model m (pair effects are common to all labels in a model and cancel in a class contrast):
  y[m, l] = class effect + a_shared[l] + a_model[m, l] + mean over P pairs of residuals e[m, l, p]
- label variance: the upper 80% bound of the pilot's estimates (3 classes x 2 conditions x 2 models; negatives at 0), split into a part
  shared across models (pilot correlation of label deviations between the two models) and a model-specific part;
- residuals e: resampled from the pilot's actual label x pair residuals (heavy tails kept), Llama-like and Phi-4-like models alternating;
- test (planned pooled primary): exact permutation test over which 8 of the 16 sensitive/salient labels are "sensitive", the same
  assignment in all five models (labels are shared), statistic = mean over models of (mean sensitive - mean salient); two-sided.
- per-model tests (secondary): the same test within each model, Holm over five.
Greedy identification (second primary): per-label rates with label variance and binomial noise over n candidates, same pooled test.
Usage: python analysis/v7_power.py   -> results/v7_power.json"""
import json, sys, importlib.util, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
spec = importlib.util.spec_from_file_location("pa", ROOT / "analysis" / "v7_pilot_analyze.py"); pa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pa)
PILOT = json.load(open(RES / "v7pilot_analysis.json", encoding="utf-8"))
MODELS_PILOT = ["llama-31-8b-instruct", "phi-4"]
N_MODELS, N_LAB = 5, 8
GREEDY_SESOI, GREEDY_MARGIN = 20, 0.10   # pp; proportion (v6's reused +-10 pp threshold for equivalence)
SESOI = MARGIN = 0.5            # nats: v6's equivalence margin and the size of v6 effects reported as meaningful (dossier §21, §23.8)
COMBOS = np.array([[1.0 if i in c else 0.0 for i in range(2 * N_LAB)] for c in itertools.combinations(range(2 * N_LAB), N_LAB)])   # 12,870 x 16

def pilot_inputs():
    lv = [max(0.0, v["label"]) for m in MODELS_PILOT for k, v in PILOT[m]["kl_components"].items() if not k.endswith("instr-none")]
    sig2_label = float(np.percentile(lv, 80))
    resid = {}
    for m in MODELS_PILOT:                                              # label x pair residuals after label and pair means, none + instr cells
        U = pa.unit_table(RES / f"v7pilot_{m}"); r = []
        for (cls, cond), g in U.groupby(["cls", "cond"]):
            Y = g.pivot(index="label", columns="pair", values="llr").dropna(axis=1).values
            r += list((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()).ravel() * np.sqrt(Y.size / ((Y.shape[0] - 1) * (Y.shape[1] - 1))))
        resid[m] = np.array(r)
    dev = {}
    for m in MODELS_PILOT:
        U = pa.unit_table(RES / f"v7pilot_{m}"); lm = U[U.cond == "none"].groupby(["cls", "label"]).llr.mean()
        dev[m] = lm - lm.groupby(level="cls").transform("mean")
    rho = float(np.corrcoef(dev[MODELS_PILOT[0]], dev[MODELS_PILOT[1]])[0, 1])
    g = [v for m in MODELS_PILOT for v in PILOT[m]["greedy_components"].values()]
    greedy_label = float(np.percentile([max(0.0, x["label"]) for x in g], 80)); greedy_mean = float(np.mean([x["mean"] for x in g]))
    return dict(sig2_label=sig2_label, rho=rho, resid=resid, greedy_label=greedy_label, greedy_mean=greedy_mean)

def perm_p(ybar, rank=False):
    """Exact two-sided permutation p for 16 label values (first 8 sensitive); rank=True: the same test on ranks (Wilcoxon-Mann-Whitney)."""
    if rank: ybar = ybar.argsort().argsort().astype(float)
    obs = ybar[:N_LAB].mean() - ybar[N_LAB:].mean(); tot = ybar.sum()
    s = COMBOS @ ybar; null = s / N_LAB - (tot - s) / N_LAB
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))

def label_ci(ybar, level):
    """t interval (df 14) for mean(sensitive) - mean(salient) over the 16 pooled label means; also the within-class label variance."""
    a, b = ybar[:N_LAB], ybar[N_LAB:]; s2w = (a.var(ddof=1) + b.var(ddof=1)) / 2; se = np.sqrt(s2w * 2 / N_LAB)
    h = stats.t.ppf(0.5 + level / 2, 2 * N_LAB - 2) * se; d = a.mean() - b.mean(); return d - h, d + h, s2w

def holm_any(ps, alpha=0.05):
    ps = sorted(ps); return [p <= alpha / (len(ps) - i) for i, p in enumerate(ps)]

def sim_kl(inp, delta, P, n_sim, rng, outlier=0.0):
    """outlier > 0: in each model, one random salient label gets an extra +outlier nats (the Phi-4 'important number' pattern)."""
    s2 = inp["sig2_label"]; sh, sp = inp["rho"] * s2, (1 - inp["rho"]) * s2
    pools = [inp["resid"][MODELS_PILOT[m % 2]] for m in range(N_MODELS)]
    eff = np.r_[np.full(N_LAB, delta / 2), np.full(N_LAB, -delta / 2)]; hit_pool = hit_any = hit_rank = 0; n_models_sig = []; ci = []
    for _ in range(n_sim):
        a_sh = rng.normal(0, np.sqrt(sh), 2 * N_LAB)
        Y = np.stack([eff + a_sh + rng.normal(0, np.sqrt(sp), 2 * N_LAB) + rng.choice(pools[m], (2 * N_LAB, P)).mean(1) for m in range(N_MODELS)])
        if outlier:
            for m in range(N_MODELS): Y[m, N_LAB + rng.integers(N_LAB)] += outlier
        yb = Y.mean(0); lo95, hi95, s2w = label_ci(yb, 0.95); lo90, hi90, _ = label_ci(yb, 0.90)
        ci.append((hi95 - lo95, lo95 <= delta <= hi95, -MARGIN < lo90 and hi90 < MARGIN, s2w))
        hit_pool += perm_p(Y.mean(0)) < 0.05; hit_rank += perm_p(Y.mean(0), rank=True) < 0.05
        sig = holm_any([perm_p(Y[m]) for m in range(N_MODELS)]); n_models_sig.append(sum(sig)); hit_any += any(sig)
    return dict(pooled=hit_pool / n_sim, pooled_rank=hit_rank / n_sim, per_model_any_holm=hit_any / n_sim, per_model_mean_sig=float(np.mean(n_models_sig)),
                ci95_halfwidth_median=float(np.median([c[0] for c in ci]) / 2), ci95_coverage=float(np.mean([c[1] for c in ci])),
                equivalent_at_margin=float(np.mean([c[2] for c in ci])), within_class_label_var_median=float(np.median([c[3] for c in ci])))

def sim_greedy(inp, delta_pp, n_cand, n_sim, rng, detail=False):
    p0, lv = inp["greedy_mean"], inp["greedy_label"]; eff = np.r_[np.full(N_LAB, delta_pp / 200), np.full(N_LAB, -delta_pp / 200)]; hit = 0; ci = []
    for _ in range(n_sim):
        a = rng.normal(0, np.sqrt(lv), 2 * N_LAB)
        Y = np.stack([rng.binomial(n_cand, np.clip(p0 + eff + a + rng.normal(0, np.sqrt(lv / 2), 2 * N_LAB), 0.01, 0.99)) / n_cand for _ in range(N_MODELS)])
        hit += perm_p(Y.mean(0)) < 0.05
        lo95, hi95, s2w = label_ci(Y.mean(0), 0.95); lo90, hi90, _ = label_ci(Y.mean(0), 0.90)
        ci.append((hi95 - lo95, lo95 <= delta_pp / 100 <= hi95, -GREEDY_MARGIN < lo90 and hi90 < GREEDY_MARGIN, s2w))
    if not detail: return hit / n_sim
    return dict(power=hit / n_sim, ci95_halfwidth_median=float(np.median([c[0] for c in ci]) / 2), ci95_coverage=float(np.mean([c[1] for c in ci])),
                equivalent_at_margin=float(np.mean([c[2] for c in ci])), within_class_label_var_median=float(np.median([c[3] for c in ci])))

def signflip_p(d, rng, n_perm=4000):
    """Two-sided sign-flip p over pair-level differences."""
    obs = abs(d.mean()); f = rng.choice([-1.0, 1.0], (n_perm, len(d))); return float((np.abs((f * d).mean(1)) >= obs - 1e-12).mean())

def sim_bridge(inp, delta, P, n_sim, rng):
    """Bridge: per model, d_p = (secret - reference number) per pair, each cell one label x pair unit (2 directions, 1 replicate);
    noise = difference of two pilot label x pair residuals (includes label x pair interaction, so not reduced by replicates).
    Per model: sign-flip over pairs, Holm over 5; 'most models' = at least 3 significant with secret > reference number. Pooled:
    mean over models per pair, sign-flip over pairs."""
    most = pooled = 0; nsig = []
    for _ in range(n_sim):
        D = np.stack([delta + rng.choice(inp["resid"][MODELS_PILOT[m % 2]], P) - rng.choice(inp["resid"][MODELS_PILOT[m % 2]], P) for m in range(N_MODELS)])
        ps = [signflip_p(D[m], rng) for m in range(N_MODELS)]; sig = holm_any(ps); order = np.argsort(ps)
        k = sum(1 for i, j in enumerate(order) if sig[i] and D[j].mean() > 0); nsig.append(k); most += k >= 3
        pooled += signflip_p(D.mean(0), rng) < 0.05
    return dict(most_models=most / n_sim, pooled=pooled / n_sim, mean_models_sig=float(np.mean(nsig)))

if __name__ == "__main__":
    rng = np.random.default_rng(20261113); inp = pilot_inputs()
    print(f"inputs: label variance (upper 80% bound of 12 pilot estimates) {inp['sig2_label']:.3f}; shared fraction rho {inp['rho']:.2f}; "
          f"residual SD Llama {inp['resid'][MODELS_PILOT[0]].std():.2f}, Phi-4 {inp['resid'][MODELS_PILOT[1]].std():.2f}; greedy label var {inp['greedy_label']:.4f}, mean {inp['greedy_mean']:.2f}")
    null = sim_kl(inp, 0.0, 24, 400, rng); print(f"null check (delta 0, P 24): pooled false-positive rate {null['pooled']:.3f}, rank {null['pooled_rank']:.3f} (target 0.05)")
    out = dict(inputs={k: v for k, v in inp.items() if k != "resid"}, null=null, kl={}, greedy={})
    for P in (24, 36):
        for d in (0.25, 0.5, 0.75, 1.0):
            r = sim_kl(inp, d, P, 400, rng); out["kl"][f"P{P}|d{d}"] = r
            print(f"KL  P={P} delta={d:.2f} nats: pooled power {r['pooled']:.2f} (rank {r['pooled_rank']:.2f}) | per-model (Holm) any {r['per_model_any_holm']:.2f}, mean models significant {r['per_model_mean_sig']:.1f}/5")
    worst = dict(inp, sig2_label=max(max(0.0, v["label"]) for m in MODELS_PILOT for k, v in PILOT[m]["kl_components"].items() if not k.endswith("instr-none")))
    out["kl_worst_case_label_variance"] = worst["sig2_label"]
    for d in (0.5, 0.75, 1.0, 1.5):                                    # worst case: the largest pilot label variance (Phi-4 salient) in every class
        r = sim_kl(worst, d, 24, 400, rng); out["kl"][f"worst|P24|d{d}"] = r
        print(f"KL  WORST label var {worst['sig2_label']:.2f}, P=24 delta={d:.2f}: pooled power {r['pooled']:.2f} (rank {r['pooled_rank']:.2f}) | per-model any {r['per_model_any_holm']:.2f}")
    for o in (2.0, 4.0):                                               # one outlying salient label per model, base label variance
        for d in (0.5, 0.75):
            r = sim_kl(inp, d, 24, 400, rng, outlier=o); out["kl"][f"outlier{o}|P24|d{d}"] = r
            print(f"KL  OUTLIER +{o:.0f} nats (one salient label per model), delta={d:.2f}: pooled power mean {r['pooled']:.2f} | rank {r['pooled_rank']:.2f}")
    print("-- CI and equivalence (typical noise, P=24): 95% t interval over the 16 pooled label means; equivalence = 90% CI inside +-0.5 nats")
    for d in (0.0, 0.25, 0.5):
        r = sim_kl(inp, d, 24, 400, rng); out["kl"][f"ci|P24|d{d}"] = r
        print(f"   delta={d:.2f}: 95% CI half-width {r['ci95_halfwidth_median']:.2f}, coverage {r['ci95_coverage']:.2f}, declared equivalent {r['equivalent_at_margin']:.2f}")
    print("-- label-variance sweep at the SESOI (0.5 nats, P=24): power and the observable within-class variance of pooled label means")
    out["sweep"] = {}
    for v in (0.05, 0.144, 0.25, 0.4, 0.6, 1.0, 2.24):
        r = sim_kl(dict(inp, sig2_label=v), SESOI, 24, 300, rng); out["sweep"][str(v)] = r
        print(f"   label var {v:.3f}: power {r['pooled']:.2f}, CI half-width {r['ci95_halfwidth_median']:.2f}, observed within-class var (median) {r['within_class_label_var_median']:.3f}")
    for n in (40, 80):
        for d in (10, 15, 20, 30):
            pw = sim_greedy(inp, d, n, 400, rng); out["greedy"][f"n{n}|d{d}"] = pw
            print(f"greedy n={n} candidates, delta={d} pp: pooled power {pw:.2f}")
    print("-- greedy (n=40): CI, equivalence at +-10 pp, and a label-variance sweep at the SESOI (20 pp)")
    for d in (0, 10, 20):
        r = sim_greedy(inp, d, 40, 300, rng, detail=True); out["greedy"][f"ci|n40|d{d}"] = r
        print(f"   delta={d} pp: power {r['power']:.2f}, 95% CI half-width {100 * r['ci95_halfwidth_median']:.1f} pp, coverage {r['ci95_coverage']:.2f}, equivalent {r['equivalent_at_margin']:.2f}")
    out["greedy_sweep"] = {}
    for v in (0.0122, 0.02, 0.03, 0.05, 0.08):
        r = sim_greedy(dict(inp, greedy_label=v), GREEDY_SESOI, 40, 300, rng, detail=True); out["greedy_sweep"][str(v)] = r
        print(f"   label var {v:.4f}: power {r['power']:.2f}, CI half-width {100 * r['ci95_halfwidth_median']:.1f} pp, observed within-class var (median) {r['within_class_label_var_median']:.4f}")
    print("-- bridge (secret vs reference number, none): per-model Holm, 'most models' (>= 3/5 significant, secret higher), pooled")
    out["bridge"] = {}
    for P in (24, 48):
        for d in (0.5, 0.75, 1.0, 1.5):
            r = sim_bridge(inp, d, P, 300, rng); out["bridge"][f"P{P}|d{d}"] = r
            print(f"   P={P} delta={d:.2f}: most-models power {r['most_models']:.2f} (mean {r['mean_models_sig']:.1f}/5 significant), pooled {r['pooled']:.2f}")
    r = sim_bridge(inp, 0.0, 24, 300, rng); out["bridge"]["P24|d0"] = r; print(f"   null P=24: most-models {r['most_models']:.3f}, pooled FPR {r['pooled']:.3f}")
    (RES / "v7_power.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
