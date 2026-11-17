"""R0 audit (analysis/r0_plan.md §1-8): exploratory, reproducible reanalysis of the v5 and v6 sampled data and the v5b/v6
greedy data. Nothing here changes a pre-registered verdict; pre-registered scripts and result files are not modified.

Usage: python analysis/r0_audit.py            -> results/r0_audit.json (+ printed summary)
Deterministic: bootstrap seed 0 (10,000 draws; 2,000 for the decomposition), sign-flip 20,000 (pre-registered function)."""
import datetime, itertools, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r0_common import (RUNS, V6_KEY, V5_KEY, RES, load, cell, pm, summ, holm, signflip, boot_ci, spearman, chi2_sf_even,
                       greedy_scores, sha256, jsonable, stratum)
sys.stdout.reconfigure(encoding="utf-8")
B_DEC = 2000; N_SHUFFLE = 200; LN = np.log

# ------------------------------------------------------------------ load
D = {name: load(name) for name in RUNS}
INSTR = {"v5": ["none", "ord", "emph", "terse"], "v6": ["none", "ref", "ord", "emph", "terse"]}
CONTRASTS = {"P1_ord-none": ("ord", "none"), "P2_emph-ord": ("emph", "ord"), "terse-none": ("terse", "none"),
             "S2_ref-none": ("ref", "none"), "none-ord": ("none", "ord"), "ref-ord": ("ref", "ord"),
             "emph-ord": ("emph", "ord"), "terse-ord": ("terse", "ord")}

# ------------------------------------------------------------------ §2 estimands: direction table per cell
def dir_table(a):
    """One row per (pair, member) direction: clean (a), all (b), pair-specific corrected (c) and sensitivities,
    stratum-4 analogues. Decoy-context answers = the other member of the same pair (their alt is this direction's s)."""
    P_C, P_Cp = a.use.mean(), a.clean_alt.mean()                                   # pooled (sensitivity 2)
    P_4, P_4p = (a.stratum == 4).mean(), (a.stratum_alt == 4).mean()
    by = {k: g for k, g in a.groupby(["pair_id", "member"])}
    rows = []
    for (p, m), g in by.items():
        o = [g2 for (p2, m2), g2 in by.items() if p2 == p and m2 != m]
        o = o[0] if o else None
        n, no = len(g), (len(o) if o is not None else 0)
        r = dict(pair_id=p, member=m, value=g.value.iat[0], n=n)
        r["clean"] = g.loc[g.use, "llr"].mean() if g.use.any() else np.nan
        r["all"] = g.llr.mean()
        k, ko = g.use.sum(), (o.clean_alt.sum() if no else 0)
        ps, pp = k / n, (ko / no if no else np.nan)
        r["undef_c"] = bool(np.isfinite(r["clean"]) and not (ps > 0 and pp > 0))
        r["cond"] = r["clean"] - (LN(ps) - LN(pp)) if (np.isfinite(r["clean"]) and ps > 0 and pp > 0) else np.nan
        r["cond_lap"] = r["clean"] - (LN((k + 1) / (n + 2)) - LN((ko + 1) / (no + 2))) if np.isfinite(r["clean"]) and no else np.nan
        r["cond_pool"] = r["clean"] - (LN(P_C) - LN(P_Cp)) if np.isfinite(r["clean"]) else np.nan
        s4 = g.stratum == 4
        r["s4"] = g.loc[s4, "llr"].mean() if s4.any() else np.nan
        k4, k4o = s4.sum(), ((o.stratum_alt == 4).sum() if no else 0)
        q, qo = k4 / n, (k4o / no if no else np.nan)
        r["undef_s4"] = bool(np.isfinite(r["s4"]) and not (q > 0 and qo > 0))
        r["s4_cond"] = r["s4"] - (LN(q) - LN(qo)) if (np.isfinite(r["s4"]) and q > 0 and qo > 0) else np.nan
        r["s4_cond_lap"] = r["s4"] - (LN((k4 + 1) / (n + 2)) - LN((k4o + 1) / (no + 2))) if np.isfinite(r["s4"]) and no else np.nan
        r["s4_pool"] = r["s4"] - (LN(P_4) - LN(P_4p)) if np.isfinite(r["s4"]) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def s4_eb(a, vals, A4):
    """Amendment A1: s4_cond with P_s(stratum 4) from an empirical-Bayes beta-binomial posterior mean fitted across the
    values in the cell (defined for every direction, including k = 0), and P_s'(stratum 4 relative to s) from the
    value-specific base rate (answers whose in-context value != s; Jeffreys). A4[j, x] = 1 if answer j is in stratum 4
    relative to value vals[x]. `a` must have a 0..n-1 index."""
    idx = {v: i for i, v in enumerate(vals)}; nv = len(vals)
    own = a.value.map(idx).values; s4own = A4[np.arange(len(a)), own].astype(float)
    n = np.bincount(own, minlength=nv).astype(float); k = np.bincount(own, weights=s4own, minlength=nv)
    base = (A4.sum(0) - k + 0.5) / (len(a) - n + 1.0)
    pres = n > 0; r = k[pres] / n[pres]
    mu = float(np.clip(r.mean(), 1e-3, 1 - 1e-3)); var = float(np.mean((r - r.mean()) ** 2)); nbar = float(n[pres].mean())
    rho = float(np.clip((var * nbar / (mu * (1 - mu)) - 1) / (nbar - 1), 1e-6, 0.999)); S = max(1.0 / rho - 1.0, 1.0)
    p = np.clip((k + mu * S) / np.maximum(n + S, 1e-12), 1e-9, 1 - 1e-9)
    rows = []
    for (pid, m), g in a.groupby(["pair_id", "member"]):
        sv = idx[g.value.iat[0]]; sel = A4[g.index.values, sv] == 1
        mean4 = float(g.llr.values[sel].mean()) if sel.any() else np.nan
        c = mean4 - (LN(p[sv]) - LN(base[sv])) if sel.any() else np.nan
        # weighted contribution of the stratum-4 conditional KL to the decomposition: P_s(stratum 4) x conditional KL
        rows.append(dict(pair_id=pid, member=m, s4_cond_eb=c, s4_p_eb=float(p[sv]), s4_contrib_eb=float(p[sv]) * c if sel.any() else np.nan))
    return pd.DataFrame(rows)

ESTIMANDS = ["clean", "all", "cond", "cond_lap", "cond_pool", "s4", "s4_cond", "s4_cond_lap", "s4_pool", "s4_cond_eb", "s4_p_eb", "s4_contrib_eb"]
def pair_level(dt, est): return dt.groupby("pair_id")[est].mean()                 # mean over available directions

def full_summ(x):
    x = pd.Series(x).dropna(); s = summ(x)
    s["ci95"] = boot_ci(x); s["ci90"] = boot_ci(x, (5, 95)); s["ci_bonf7"] = boot_ci(x, (100 * 0.05 / 14, 100 - 100 * 0.05 / 14))
    s["sd"] = float(x.std(ddof=1)); return s

OUT = {"meta": dict(script="analysis/r0_audit.py", plan="analysis/r0_plan.md", date=str(datetime.date.today()),
                    bootstrap_draws=10000, decomposition_draws=B_DEC, shuffles=N_SHUFFLE, seed=0, signflip_draws=20000,
                    amendment_A1=dict(primary_beyond_literal="decomposition residual contrasts (r0_plan Amendment A1)",
                                      descriptive_only=["s4", "s4_cond", "s4_cond_lap", "s4_pool"],
                                      s4_cond_eb="descriptive: in simulation it met the bias criterion but not coverage (0.85-0.91) or the null false-positive limit (0.08); decision rule fixed in A1 before the run",
                                      null_floor_fields="label-shuffle reference: overstates the floor (simulated floor about 0.05 nats); "
                                                        "containment levels are upward-biased near 0, interpret contrasts",
                                      coverage="pair-bootstrap percentile CIs covered 0.91-0.96 in simulation; sign-flip p are the tests of record")),
       "self_check": {}, "estimands": {}, "rates": {}, "decomposition": {}, "equivalence": {}, "priming": {},
       "cross_model": {}, "greedy": {}, "three_layer": {}, "S3": {}, "provenance": {}}
DT = {}
for name, (d, ver) in D.items():
    T = {}
    for i in INSTR[ver]:
        a = d[d.cell == cell(ver, i)].reset_index(drop=True)
        vals = sorted(set(a.value) | set(a.alt))
        A4 = np.array([[stratum(t, x, nn) == 4 for x in vals] for t, nn in zip(a.response, a.nums)], float)
        T[i] = dir_table(a).merge(s4_eb(a, vals, A4), on=["pair_id", "member"], how="left")
    DT[name] = T

# ------------------------------------------------------------------ §1 self-checks
v6j = json.load(open(RES / "v6_analysis.json")); v5j = json.load(open(RES / "v5_analysis.json"))
checks, fails = [], []
def chk(label, mine, ref, tol=1e-9):
    ok = abs(mine - ref) <= tol; checks.append(dict(check=label, mine=mine, reference=ref, ok=bool(ok)))
    if not ok: fails.append(label)
for name, (d, ver) in D.items():
    u = d[d.use]; kl = lambda c, u=u: pm(u[u.cell == c], "llr")
    for i in INSTR[ver]:                                                             # direction table == pre-registered pm
        a = pair_level(DT[name][i], "clean"); b = kl(cell(ver, i))
        chk(f"{name} clean pair-level == pm ({i})", float((a - b).abs().max()), 0.0, 1e-12)
    if ver == "v6":
        R = v6j["per_model"][V6_KEY[name]]
        chk(f"{name} P1", summ(kl("list|ord") - kl("list|none"))["estimate"], R["P1"]["estimate"])
        chk(f"{name} P2", summ(kl("list|emph") - kl("list|ord"))["estimate"], R["P2"]["estimate"])
        for c in ["none", "ref", "emph", "terse"]:
            chk(f"{name} dKL {c}-ord", summ(kl(f"list|{c}") - kl("list|ord"))["estimate"], R["dKL_vs_ord"][c]["estimate"])
        chk(f"{name} S2 KL", summ(kl("list|ref") - kl("list|none"))["estimate"], R["S2_kl_ref_minus_none"]["estimate"])
        gap = lambda c: 100 * (pm(d[d.cell == c], "has_s") - pm(d[d.cell == c], "has_d"))
        chk(f"{name} S2 gap", summ(gap("list|ref") - gap("list|none"))["estimate"], R["S2_gap_ref_minus_none"]["estimate"])
    else:
        R = v5j[V5_KEY[name]]
        chk(f"{name} v5 C1", summ(kl("emph|T1.0") - kl("ord|T1.0"))["estimate"], R["C1"]["estimate"])
        allkl = lambda c: pm(d[d.cell == c], "llr")
        chk(f"{name} v5 all_answers C1", summ(allkl("emph|T1.0") - allkl("ord|T1.0"))["estimate"], R["all_answers"]["C1"]["estimate"])
OUT["self_check"] = dict(passed=not fails, n=len(checks), failed=fails, checks=checks)
print(f"SELF-CHECK: {len(checks) - len(fails)}/{len(checks)} passed")
if fails:
    print("FAILED:", fails); Path(RES / "r0_audit.json").write_text(json.dumps(jsonable(OUT), indent=1)); sys.exit(1)

# ------------------------------------------------------------------ §2-3 contrasts under every estimand
for name, (d, ver) in D.items():
    E = {}
    for cname, (c1, c2) in CONTRASTS.items():
        if c1 not in INSTR[ver] or c2 not in INSTR[ver]: continue
        E[cname] = {}
        for est in ESTIMANDS:
            x = (pair_level(DT[name][c1], est) - pair_level(DT[name][c2], est)).dropna()
            E[cname][est] = full_summ(x) if len(x) > 1 else None
    undefined = {i: dict(cond_directions=int(t.undef_c.sum()), s4_directions=int(t.undef_s4.sum()),
                         cond_pairs_all_undefined=int(t.groupby("pair_id").cond.apply(lambda s: s.isna().all()).sum()),
                         s4_pairs_all_undefined=int(t.groupby("pair_id").s4_cond.apply(lambda s: s.isna().all()).sum()))
                 for i, t in DT[name].items()}
    levels = {i: {est: float(pair_level(t, est).mean()) for est in ESTIMANDS} for i, t in DT[name].items()}
    OUT["estimands"][name] = dict(contrasts=E, undefined=undefined, cell_levels=levels)

# ------------------------------------------------------------------ §8 rates per cell
for name, (d, ver) in D.items():
    R = {}
    for c, a in d.groupby("cell"):
        R[c] = dict(n=len(a), structural=float(a.structural.mean()), verbal=float(a.verbal.mean()), digit=float(a.digit.mean()),
                    **{f"stratum_{k}": float((a.stratum == k).mean()) for k in [1, 2, 3, 4]},
                    refusal=float(a.refusal.mean()), usable_A2=float((a.use & (a.n_nums >= 10)).mean()))
    OUT["rates"][name] = R

# ------------------------------------------------------------------ §4 decomposition (chain rule, per pair)
def kl_bern(p, q): return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

class Decomp:
    """Per-cell containment / residual decomposition with value-specific base rates and EB-shrunk emission rates."""
    def __init__(self, a, pairs):
        self.vals = sorted(set(a.value) | set(a.alt)); idx = {v: i for i, v in enumerate(self.vals)}; nv = len(self.vals)
        self.M = np.array([[v in set(L) for v in self.vals] for L in a.nums], float)          # answers x values
        self.own = a.value.map(idx).values; self.alt = a.alt.map(idx).values
        pidx = {p: i for i, p in enumerate(pairs)}; self.pa = a.pair_id.map(pidx).values; self.P = len(pairs)
        dkey = (a.pair_id + "|" + a.member).values; dirs = sorted(set(dkey)); di = {k: i for i, k in enumerate(dirs)}
        self.da = np.array([di[k] for k in dkey]); self.nd = len(dirs)
        first = a.drop_duplicates(["pair_id", "member"]).set_index(a.drop_duplicates(["pair_id", "member"]).pair_id + "|" + a.drop_duplicates(["pair_id", "member"]).member)
        self.ds = np.array([idx[first.loc[k, "value"]] for k in dirs]); self.dsp = np.array([idx[first.loc[k, "alt"]] for k in dirs])
        self.dp = np.array([pidx[first.loc[k, "pair_id"]] for k in dirs])
        self.dtot = np.bincount(self.da, weights=a.llr.values, minlength=self.nd) / np.bincount(self.da, minlength=self.nd)
        val_pair = np.full(nv, -1)
        for s, p in zip(self.own, self.pa): val_pair[s] = p
        self.val_pair = val_pair
        self.has_s = a.has_s.values; self.has_d = a.has_d.values
    def contain(self, wp, own=None, scale=1.0):
        """wp: (B, P) pair weights. Returns per-direction containment (B, nd)."""
        own = self.own if own is None else own
        ar = np.arange(len(own)); m_own = self.M[ar, own]; nv = len(self.vals)
        O = np.zeros((len(own), nv)); O[ar, own] = 1.0
        wa = wp[:, self.pa]                                                                      # (B, n_ans)
        num = wa @ self.M - (wa * m_own) @ O; den = wa.sum(1, keepdims=True) - wa @ O
        beta = (num + 0.5) / (den + 1.0)
        k = O.T @ m_own; n = O.sum(0); present = n > 0
        r = np.where(present, k / np.maximum(n, 1), 0.0)
        mult = np.where(present[None, :], wp[:, np.maximum(self.val_pair, 0)], 0.0)
        W = mult.sum(1, keepdims=True); mu = (mult @ r)[:, None] / W
        var = (mult @ r ** 2)[:, None] / W - mu ** 2
        mu = np.clip(mu, 1e-3, 1 - 1e-3); nbar = n[present].mean()
        rho = np.clip((var * nbar / (mu * (1 - mu)) - 1) / (nbar - 1), 1e-6, 0.999)
        S = np.maximum(1.0 / rho - 1.0, 1.0) * scale
        eps = (k[None, :] + mu * S) / (n[None, :] + S)
        eps = np.clip(eps, 1e-9, 1 - 1e-9); beta = np.clip(beta, 1e-9, 1 - 1e-9)
        ds = self.ds if own is self.own else self._ds_override; dsp = self.dsp if own is self.own else self._dsp_override
        bi = np.arange(wp.shape[0])[:, None]
        return kl_bern(eps[bi, ds[None, :]], beta[bi, ds[None, :]]) + kl_bern(beta[bi, dsp[None, :]], eps[bi, dsp[None, :]])
    def aggregate(self, wp, per_dir):
        """Pair-weighted mean of pair means (both directions)."""
        pairsum = np.zeros((wp.shape[0], self.P)); cnt = np.zeros(self.P)
        np.add.at(pairsum.T, self.dp, per_dir.T); np.add.at(cnt, self.dp, 1)
        pm_ = pairsum / np.maximum(cnt, 1); w = wp * (cnt > 0)
        return (w * pm_).sum(1) / w.sum(1)
    def pooled(self, first=None):
        hs, hd = self.has_s.astype(int), self.has_d.astype(int)
        if first is None:
            q = np.bincount(hs * 2 + hd, minlength=4) + 0.5; p = q / q.sum(); sw = p[[0, 2, 1, 3]]
        else:
            fs, fd = first
            code = hs * 8 + hd * 4 + fs * 2 + fd
            q = np.bincount(code, minlength=16) + 0.5; p = q / q.sum()
            perm = [((c >> 2) & 1) * 8 + ((c >> 3) & 1) * 4 + (c & 1) * 2 + ((c >> 1) & 1) for c in range(16)]
            sw = p[perm]
        return float(np.sum(p * np.log(p / sw)))
    def mi(self):
        j = np.bincount(self.has_s.astype(int) * 2 + self.has_d.astype(int), minlength=4) / len(self.has_s)
        ps = np.array([j[0] + j[1], j[2] + j[3]]); pd_ = np.array([j[0] + j[2], j[1] + j[3]])
        return float(sum(j[a * 2 + b] * np.log(j[a * 2 + b] / (ps[a] * pd_[b])) for a in (0, 1) for b in (0, 1) if j[a * 2 + b] > 0))
    def null_floor(self, rng):
        vals = []
        for _ in range(N_SHUFFLE):
            flip = rng.random(self.nd) < 0.5
            own = np.where(flip[self.da], self.alt, self.own)
            self._ds_override = np.where(flip, self.dsp, self.ds); self._dsp_override = np.where(flip, self.ds, self.dsp)
            c = self.contain(np.ones((1, self.P)), own=own.copy())
            vals.append(float(self.aggregate(np.ones((1, self.P)), c)[0]))
        return float(np.mean(vals)), float(np.percentile(vals, 95))

rng_boot = np.random.default_rng(0)
for name, (d, ver) in D.items():
    pairs = sorted(d.pair_id.unique()); P = len(pairs)
    WP = np.zeros((B_DEC, P))
    for b in range(B_DEC): WP[b] = np.bincount(rng_boot.integers(0, P, P), minlength=P)
    one = np.ones((1, P)); res = {}; draws = {}
    for i in INSTR[ver]:
        a = d[d.cell == cell(ver, i)].reset_index(drop=True)
        dc = Decomp(a, pairs)
        tot_pt = float(dc.aggregate(one, dc.dtot[None, :])[0]); tot_bs = dc.aggregate(WP, np.repeat(dc.dtot[None, :], B_DEC, 0))
        out = dict(total_all_answers=tot_pt)
        for scale in [1.0, 0.5, 2.0]:
            c_pt = float(dc.aggregate(one, dc.contain(one, scale=scale))[0])
            tag = "" if scale == 1.0 else f"_prior_x{scale}"
            out[f"containment{tag}"] = c_pt; out[f"residual{tag}"] = tot_pt - c_pt
            if scale == 1.0:
                c_bs = dc.aggregate(WP, dc.contain(WP)); draws[i] = (tot_bs, c_bs)
                out["containment_ci95"] = [float(v) for v in np.percentile(c_bs, [2.5, 97.5])]
                out["residual_ci95"] = [float(v) for v in np.percentile(tot_bs - c_bs, [2.5, 97.5])]
                out["containment_share"] = c_pt / tot_pt if tot_pt else float("nan")
        out["pooled_lower_bound"] = dc.pooled()
        fs = a.first_s.fillna(0).astype(int).values; fd = a.first_d.fillna(0).astype(int).values
        out["pooled_first_position_lower_bound"] = dc.pooled((fs, fd))
        out["null_floor_mean"], out["null_floor_p95"] = dc.null_floor(np.random.default_rng(1))
        out["pooled_mutual_information_s_d"] = dc.mi()
        res[i] = out
    con = {}
    for cname, (c1, c2) in CONTRASTS.items():
        if c1 not in draws or c2 not in draws: continue
        dt = draws[c1][0] - draws[c2][0]; dc_ = draws[c1][1] - draws[c2][1]
        con[cname] = dict(d_total=res[c1]["total_all_answers"] - res[c2]["total_all_answers"],
                          d_containment=res[c1]["containment"] - res[c2]["containment"],
                          d_residual=res[c1]["residual"] - res[c2]["residual"],
                          d_total_ci95=[float(v) for v in np.percentile(dt, [2.5, 97.5])],
                          d_containment_ci95=[float(v) for v in np.percentile(dc_, [2.5, 97.5])],
                          d_residual_ci95=[float(v) for v in np.percentile(dt - dc_, [2.5, 97.5])])
    OUT["decomposition"][name] = dict(cells=res, contrasts=con)

# ------------------------------------------------------------------ §5 equivalence
def classify(ci, m):
    lo, hi = ci
    if -m <= lo and hi <= m: return "equivalent"
    if lo > m or hi < -m: return "meaningfully different"
    return "inconclusive"
EQ = {"terse-none_KL_clean": {}, "S2_containment_gap_ref-none_pp": {}}
for name, (d, ver) in D.items():
    s = OUT["estimands"][name]["contrasts"]["terse-none"]["clean"]
    EQ["terse-none_KL_clean"][name] = dict(estimate=s["estimate"], ci90=s["ci90"], excludes_zero=bool(s["ci90"][0] > 0 or s["ci90"][1] < 0),
                                           margins={str(m): classify(s["ci90"], m) for m in [0.25, 0.5, 1.0]},
                                           margin_status="post hoc (no pre-specified margin)")
    if ver == "v6":
        gap = lambda c: 100 * (pm(d[d.cell == c], "has_s") - pm(d[d.cell == c], "has_d"))
        x = (gap("list|ref") - gap("list|none")).dropna(); s = summ(x); ci = boot_ci(x, (5, 95))
        EQ["S2_containment_gap_ref-none_pp"][name] = dict(estimate=s["estimate"], ci90=ci, excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                                                          margins={str(m): classify(ci, m) for m in [5, 10, 15]},
                                                          margin_status="10 pp = v6 S1a threshold reused as exploratory benchmark")
OUT["equivalence"] = EQ

# ------------------------------------------------------------------ §6 priming across models
rows = []
for name, (d, ver) in D.items():
    gap = 100 * (pm(d[d.cell == cell(ver, "none")], "has_s") - pm(d[d.cell == cell(ver, "none")], "has_d"))
    lv = OUT["estimands"][name]["cell_levels"]
    rows.append(dict(model=name, gap_none=float(gap.mean()), KL_ord=lv["ord"]["clean"], KL_none=lv["none"]["clean"],
                     P1=OUT["estimands"][name]["contrasts"]["P1_ord-none"]["clean"]["estimate"]))
PR = pd.DataFrame(rows)
rho = spearman(PR.gap_none, PR.KL_ord)
null = np.array([spearman(PR.gap_none, PR.KL_ord.values[list(p)]) for p in itertools.permutations(range(len(PR)))])
coup_g = np.array([spearman(PR.gap_none, PR.KL_ord.values[list(p)] - PR.KL_none.values) for p in itertools.permutations(range(len(PR)))])
coup_k = np.array([spearman(PR.KL_none, PR.KL_ord.values[list(p)] - PR.KL_none.values) for p in itertools.permutations(range(len(PR)))])
OUT["priming"] = dict(
    table=rows, n_models=len(PR), note="n = 7 models: low power",
    main_spearman_gap_none_vs_KL_ord=dict(rho=rho, p_exact_two_sided=float(np.mean(np.abs(null) >= abs(rho) - 1e-12)), permutations=len(null)),
    descriptive_coupled=dict(
        spearman_gap_none_vs_P1=spearman(PR.gap_none, PR.P1), spearman_KL_none_vs_P1=spearman(PR.KL_none, PR.P1),
        coupling_reference_gap=dict(mean=float(coup_g.mean()), p2_5=float(np.percentile(coup_g, 2.5)), p97_5=float(np.percentile(coup_g, 97.5))),
        coupling_reference_KLnone=dict(mean=float(coup_k.mean()), p2_5=float(np.percentile(coup_k, 2.5)), p97_5=float(np.percentile(coup_k, 97.5))),
        note="coupled statistics share the baseline term; reference distributions come from permuting ord levels across models"))

# ------------------------------------------------------------------ §7 cross-model P1 table and Cochran Q
p1 = {name: OUT["estimands"][name]["contrasts"]["P1_ord-none"]["clean"] for name in D}
hp = holm([p1[n]["p"] for n in D])
th = np.array([p1[n]["estimate"] for n in D]); se = np.array([p1[n]["sd"] / np.sqrt(p1[n]["n"]) for n in D]); w = 1 / se ** 2
tbar = float((w * th).sum() / w.sum()); Q = float((w * (th - tbar) ** 2).sum()); dfq = len(th) - 1
OUT["cross_model"] = dict(
    P1_table={n: dict(estimate=p1[n]["estimate"], ci95_pointwise=p1[n]["ci95"], ci_simultaneous_bonferroni7=p1[n]["ci_bonf7"],
                      p_signflip=p1[n]["p"], p_holm7=hp[k], pairs_pos=p1[n]["pairs_pos"], n_pairs=p1[n]["n"]) for k, n in enumerate(D)},
    cochran=dict(Q=Q, df=dfq, p=chi2_sf_even(Q, dfq), I2=max(0.0, (Q - dfq) / Q), fixed_effect_mean=tbar,
                 formula="theta_i = P1 (clean); SE_i = SD(pair contrasts)/sqrt(n_i); w_i = 1/SE_i^2; Q = sum w_i (theta_i - theta_bar)^2",
                 assumptions="independent models, SEs treated as known; heterogeneity among the 7 tested models only"))

# ------------------------------------------------------------------ §7 greedy replay: ord - none, three-layer table
G = {}; gchk = []
for name, (folder, ver, gfolder) in RUNS.items():
    instrs = ["none", "ord", "emph", "terse"] + (["ref"] if ver == "v6" else [])
    G[name] = {i: greedy_scores(gfolder, i)[0] for i in instrs}
    if ver == "v6":
        ref = v6j["per_model"][V6_KEY[name]]["greedy"]
        for i in instrs:
            chk(f"{name} greedy replay {i}", 100 * float(np.mean(list(G[name][i].values()))), ref[i]["replay"], 1e-9)
def paired(name, i1, i2):
    common = sorted(set(G[name][i1]) & set(G[name][i2])); x = np.array([G[name][i1][v] - G[name][i2][v] for v in common])
    ci = boot_ci(100 * x)
    return dict(diff_pp=100 * float(x.mean()), ci95=ci, p_signflip=float(signflip(pd.Series(x))) if np.any(x != 0) else 1.0, n_candidates=len(common))
gr = {n: paired(n, "ord", "none") for n in D}
hg = holm([gr[n]["p_signflip"] for n in D])
for k, n in enumerate(D): gr[n]["p_holm7"] = hg[k]; gr[n]["replay_none"] = 100 * float(np.mean(list(G[n]["none"].values()))); gr[n]["replay_ord"] = 100 * float(np.mean(list(G[n]["ord"].values())))
OUT["greedy"] = dict(ord_minus_none=gr, threat_model="closed set of 80 candidates; attacker knows model, prompt format and instruction; "
                     "80 greedy single-call references; observes one batched greedy answer; candidates whose answer discloses are excluded")
for name, (folder, ver, gfolder) in RUNS.items():
    if ver != "v6": continue
    L1 = np.array(list(G[name]["ref"].values()))
    OUT["three_layer"][name] = dict(L1_ref_replay_pct=100 * float(L1.mean()), L1_ci95=boot_ci(100 * L1), n_candidates=len(L1),
                                    L2_none_minus_ref=paired(name, "none", "ref"), L3_ord_minus_none=paired(name, "ord", "none"))
# greedy self-checks appended above; re-evaluate
OUT["self_check"] = dict(passed=not fails, n=len(checks), failed=fails, checks=checks)

# ------------------------------------------------------------------ S3 from the pre-registered output
rows = v6j["S3"]["rows"]
OUT["S3"] = dict(rows=rows, n_rows=len(rows), valid=sum(r["valid"] for r in rows), same_sign_valid=sum(r["same_sign"] for r in rows if r["valid"]),
                 rep_significant_valid=sum(r["p_rep"] < 0.05 for r in rows if r["valid"]),
                 note="secondary-task sample; supportive evidence, not a broad replication")

# ------------------------------------------------------------------ provenance
FIELDS = ["model_id", "model_revision", "weights_source", "load_from", "quantization", "compute_dtype", "torch", "transformers", "gpu",
          "n_gpus", "seed", "pair_seed", "prompt_hash", "prompt_version", "temp", "temps", "n_responses"]
for name, (folder, ver, gfolder) in RUNS.items():
    pv = {}
    for f in sorted({folder, gfolder}):
        meta = json.load(open(RES / f / "metadata.json"))
        fields = {}
        for k in FIELDS:
            if k in meta:
                v = meta[k]
                status = "unknown (recorded as None)" if v is None else "recorded"
                if k == "model_revision" and v is None: status = "unknown; recoverable only via Phase 1a (commit history / score reproduction)"
                fields[k] = dict(value=v, status=status)
        files = {fn: sha256(RES / f / fn) for fn in ["responses.jsonl", "scores.jsonl", "greedy_single.jsonl", "greedy_batched.jsonl"] if (RES / f / fn).exists()}
        pv[f] = dict(metadata=fields, sha256=files)
    OUT["provenance"][name] = pv
OUT["meta"]["inputs_sha256"] = {f: v["sha256"] for n in OUT["provenance"] for f, v in OUT["provenance"][n].items()}

Path(RES / "r0_audit.json").write_text(json.dumps(jsonable(OUT), indent=1), encoding="utf-8")

# ------------------------------------------------------------------ printed summary
f2 = lambda s: "—" if s is None else f"{s['estimate']:+.2f} [{s['ci95'][0]:+.2f},{s['ci95'][1]:+.2f}] p={s['p']:.4f}"
print(f"SELF-CHECK (incl. greedy): {len(checks) - len(fails)}/{len(checks)} passed" + ("" if not fails else f"  FAILED {fails}"))
print("\n=== P1 (ord − none) under each estimand")
for n in D:
    E = OUT["estimands"][n]["contrasts"]["P1_ord-none"]
    print(f"{n:18s} clean {f2(E['clean'])} | all {f2(E['all'])} | cond {f2(E['cond'])} | s4_cond {f2(E['s4_cond'])} | s4_cond_eb {f2(E['s4_cond_eb'])}")
print("\n=== S2 (ref − none), v6")
for n in D:
    E = OUT["estimands"][n]["contrasts"].get("S2_ref-none")
    if E: print(f"{n:18s} clean {f2(E['clean'])} | all {f2(E['all'])} | cond {f2(E['cond'])} | s4_cond {f2(E['s4_cond'])} | s4_cond_eb {f2(E['s4_cond_eb'])}")
print("\n=== Cross-model P1:", {n: (round(v['estimate'], 2), [round(x, 2) for x in v['ci_simultaneous_bonferroni7']], round(v['p_holm7'], 4)) for n, v in OUT['cross_model']['P1_table'].items()})
c = OUT["cross_model"]["cochran"]; print(f"Cochran Q = {c['Q']:.1f}, df = {c['df']}, p = {c['p']:.2e}, I2 = {100 * c['I2']:.0f}%")
print("\n=== Greedy replay ord − none (pp):", {n: (round(v['diff_pp'], 1), round(v['p_signflip'], 4), round(v['p_holm7'], 4)) for n, v in gr.items()})
print("\n=== Decomposition (list cells): total | containment [CI] | residual | pooled LB | null floor")
for n in D:
    for i, r in OUT["decomposition"][n]["cells"].items():
        print(f"{n:18s} {i:5s} {r['total_all_answers']:+6.2f} | {r['containment']:.3f} [{r['containment_ci95'][0]:.3f},{r['containment_ci95'][1]:.3f}] "
              f"| {r['residual']:+6.2f} | LB {r['pooled_lower_bound']:.3f} | null {r['null_floor_mean']:.3f} | MI {r['pooled_mutual_information_s_d']:.4f}")
print("\n=== Equivalence")
for k, v in EQ.items():
    for n, e in v.items(): print(f"{k:32s} {n:18s} {e['estimate']:+.2f} CI90 [{e['ci90'][0]:+.2f},{e['ci90'][1]:+.2f}] {e['margins']}")
pp = OUT["priming"]; print(f"\n=== Priming: Spearman(gap_none, KL_ord) = {pp['main_spearman_gap_none_vs_KL_ord']['rho']:+.2f}, exact p = {pp['main_spearman_gap_none_vs_KL_ord']['p_exact_two_sided']:.3f}; "
      f"coupled (desc.) gap~P1 {pp['descriptive_coupled']['spearman_gap_none_vs_P1']:+.2f} (coupling ref mean {pp['descriptive_coupled']['coupling_reference_gap']['mean']:+.2f})")
print("\nwrote", RES / "r0_audit.json")
