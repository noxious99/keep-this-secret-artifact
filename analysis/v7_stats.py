"""v7 statistics (preregistration_v7.md §4.2, §5): shared by v7_alpha.py and v7_analyze.py.

- crossed(Z, level): the crossed analysis of a class contrast from a 16 x n table (rows: 8 labels of class A, then 8 of class B;
  columns: pairs or candidates; values pooled over models). Returns estimate, SE, interval, two-sided p (t, 14 df), s2_w, sigma2_g.
- tost_p(est, se, margin): equivalence p (max of the two one-sided t p values, 14 df).
- perm_p / rank: label-level exact permutation test over the C(16, 8) class assignments (robustness).
- signflip_p(d): two-sided sign-flip p over units (pairs or candidates).
- graph_test(p, w, G, alpha): the graphical multiple-testing procedure (Bretz et al. 2009, Algorithm 1)."""
import itertools
import numpy as np
from scipy import stats

N_LAB = 8
DF = 2 * N_LAB - 2
COMBOS = np.array([[1.0 if i in c else 0.0 for i in range(2 * N_LAB)] for c in itertools.combinations(range(2 * N_LAB), N_LAB)])

def crossed(Z, level=0.95, na=N_LAB):
    """Rows 0..na-1 = class A labels, the rest class B (8 and 8 in v7; 7/8 only in leave-one-label-out). df = na + nb - 2."""
    Z = np.asarray(Z, float); n = Z.shape[1]; nb = Z.shape[0] - na; df = na + nb - 2; yb = Z.mean(1); a, b = yb[:na], yb[na:]
    s2w = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / df; est = a.mean() - b.mean(); res = []
    for blk in (Z[:na], Z[na:]):
        r = blk - blk.mean(1, keepdims=True) - blk.mean(0, keepdims=True) + blk.mean(); res.append((r ** 2).sum() / ((blk.shape[0] - 1) * (n - 1)))
    dp = Z[:na].mean(0) - Z[na:].mean(0); s2g = max(0.0, dp.var(ddof=1) - np.mean(res) * (1 / na + 1 / nb))
    se = float(np.sqrt(s2w * (1 / na + 1 / nb) + s2g / n)); h = stats.t.ppf(0.5 + level / 2, df) * se
    p = float(2 * stats.t.sf(abs(est) / se, df)) if se > 0 else (0.0 if est else 1.0)
    return dict(est=float(est), se=se, lo=float(est - h), hi=float(est + h), p=p, s2_w=float(s2w), sigma2_g=float(s2g), df=df)

def tost_p(est, se, margin, df=DF):
    if se <= 0: return 0.0 if abs(est) < margin else 1.0
    return float(max(stats.t.sf((est + margin) / se, df), stats.t.cdf((est - margin) / se, df)))

def perm_p(ybar, rank=False):
    ybar = np.asarray(ybar, float)
    if rank: ybar = ybar.argsort().argsort().astype(float)
    obs = ybar[:N_LAB].mean() - ybar[N_LAB:].mean(); tot = ybar.sum(); s = COMBOS @ ybar; null = s / N_LAB - (tot - s) / N_LAB
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))

def signflip_p(d, n_perm=20000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); obs = abs(d.mean())
    f = rng.choice([-1.0, 1.0], (n_perm, len(d))); return float(((np.abs((f * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))

def graph_test(p, w, G, alpha=0.05):
    """Bretz et al. (2009) Algorithm 1. p: dict name -> p value; w: initial weights (sum <= 1); G: transition weights G[j][l].
    Returns the set of rejected names. Strong FWER control at alpha for any p values valid under their nulls (Bonferroni-based)."""
    names = list(p); w = {k: float(w.get(k, 0.0)) for k in names}; G = {j: {l: float(G.get(j, {}).get(l, 0.0)) for l in names} for j in names}
    active, rejected = set(names), set()
    while True:
        cand = [j for j in active if p[j] <= w[j] * alpha]
        if not cand: return rejected
        j = min(cand, key=lambda k: p[k]); active.discard(j); rejected.add(j)
        for l in active: w[l] += w[j] * G[j][l]
        newG = {l: {} for l in active}
        for l in active:
            for k in active:
                if l == k: newG[l][k] = 0.0; continue
                den = 1 - G[l][j] * G[j][l]
                newG[l][k] = (G[l][k] + G[l][j] * G[j][k]) / den if den > 1e-12 else 0.0
        for l in active: G[l] = {**{k: 0.0 for k in names}, **newG[l]}
        w[j] = 0.0

def v7_graph(fam_b, alpha=0.05, alpha_b=0.025, alpha_p=0.025, eps=1e-6):
    """preregistration_v7.md §5.0 with the gatekept P3 (sensitive - neutral, KL):
      family B: Holm over its hypotheses (alpha_b split equally); when all are rejected, alpha passes to P1 and P2;
      P1, P2: alpha_p / 2 each; P3: initial weight 0.
      P1 -> P3 (P3 is tested only with alpha released by P1, or later by P2);
      P2 -> P1 and P3 equally (Holm-like recycling within P);
      P3 -> P1 and P2 equally;
      when all of P1-P3 are rejected, alpha passes to family B (epsilon edges)."""
    P = ["P1", "P2", "P3"]; w = {h: alpha_b / alpha / len(fam_b) for h in fam_b}; w.update(P1=alpha_p / alpha / 2, P2=alpha_p / alpha / 2, P3=0.0)
    G = {}
    for h in fam_b:
        same = [x for x in fam_b if x != h]; G[h] = {x: (1 - eps) / len(same) for x in same} if same else {}
        for x in ("P1", "P2"): G[h][x] = (eps if same else 1.0) / 2
    toB = {x: eps / len(fam_b) for x in fam_b}
    G["P1"] = {"P3": 1 - eps, **toB}
    G["P2"] = {"P1": (1 - eps) / 2, "P3": (1 - eps) / 2, **toB}
    G["P3"] = {"P1": (1 - eps) / 2, "P2": (1 - eps) / 2, **toB}
    return w, G

def two_family_graph(fam_a, fam_b, alpha_a, alpha_b, alpha=0.05, eps=1e-6):
    """Holm within each family (equal weights, alpha passed on within the family when a hypothesis is rejected); when a family is
    completely rejected its alpha passes to the other family (epsilon edges). Returns (w, G) for graph_test."""
    w, G = {}, {}
    for fam, other, a in ((fam_a, fam_b, alpha_a), (fam_b, fam_a, alpha_b)):
        for h in fam:
            w[h] = a / alpha / len(fam); same = [x for x in fam if x != h]
            G[h] = {x: (1 - eps) / len(same) for x in same} if same else {}
            for x in other: G[h][x] = (eps if same else 1.0) / len(other)
    return w, G
