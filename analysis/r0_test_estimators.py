"""Verification of the R0 estimators (analysis/r0_plan.md §2, §4, §9): parts 2 and 3 of the R0 formula check.

Part 3, unit tests: hand-computed cases and exact enumeration on tiny models, run against the SAME functions used by
analysis/r0_audit.py and analysis/r0_nearest_ref.py (extracted from their source with `ast`, without running them).
  - identities (1) clean-conditional correction and (2) KL chain rule, by exact enumeration of a tiny model;
  - (3) KL of independent products = sum of marginal KLs; joint convexity (pooled lower bound);
  - dir_table / pair_level on a hand-computed example, including an undefined correction;
  - Decomp against an independent re-implementation; pooled estimate by hand;
  - exact-occurrence strata; nearest-reference similarity, ties, exclusions, S matrix, permutation test.

Part 2, simulation with known truth: a toy "model" whose answer probabilities are exactly computable (20 i.i.d. draws
from a context-dependent distribution over the 80 values, or a 2-component mixture for strong dependence). The real
design is replicated (36 pairs x 2 directions x 3 answers, two cells A and B) and the real estimators are applied.
Truth is the superpopulation mean over random pair-directions: exact for total KL (i.i.d.) and containment, Monte Carlo
from the definitions for the conditional quantities.

Pass criteria, fixed before running (stated in the previous review round):
  |bias| <= 0.05 nats (10% of the smallest reported contrast, 0.5 nats); 95% CI coverage in [0.92, 0.98];
  null false-positive rate (p < .05) <= 0.07 for the KL contrasts, judged on the
  'null_contrast' scenario (two identical informative cells; the all-zero 'null' only checks the no-information floor). The containment estimator is EXPECTED to show a
  positive bias under the null (its documented floor); the simulation quantifies it.

Usage: python analysis/r0_test_estimators.py [n_reps]   (default 300)  -> results/r0_test_estimators.json"""
import ast, datetime, itertools, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from r0_common import RES, signflip, stratum, jsonable
sys.stdout.reconfigure(encoding="utf-8")

def extract(path, names, glb):
    """Compile only the named top-level definitions of a script (plus try-blocks if 'TRY' is named) into a namespace."""
    tree = ast.parse(path.read_text(encoding="utf-8")); keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names: keep.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in names for t in node.targets): keep.append(node)
        elif isinstance(node, ast.Try) and "TRY" in names: keep.append(node)
    ns = dict(glb); exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    missing = [n for n in names if n != "TRY" and n not in ns]; assert not missing, missing
    return ns
AU = extract(HERE / "r0_audit.py", {"dir_table", "pair_level", "kl_bern", "Decomp", "s4_eb", "LN", "N_SHUFFLE"}, {"np": np, "pd": pd})
NR = extract(HERE / "r0_nearest_ref.py", {"sim_positional", "sim_prefix", "sim_jaccard", "sim_lev", "sim_length", "METRICS",
                                          "best_set", "group_eval", "perm_test", "N_PERM", "TRY"}, {"np": np, "pd": pd})
dir_table, pair_level, kl_bern, Decomp, s4_eb = AU["dir_table"], AU["pair_level"], AU["kl_bern"], AU["Decomp"], AU["s4_eb"]

# ====================================================================== PART 3: unit tests
TESTS = []
def check(name, ok, detail=""):
    TESTS.append(dict(test=name, ok=bool(ok), detail=str(detail)))
def close(a, b, tol=1e-9): return abs(a - b) <= tol

# ---- identities (1), (2) by exact enumeration: alphabet 4, length 3, two contexts
rng = np.random.default_rng(123)
for trial in range(5):
    qs, qp = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    seqs = list(itertools.product(range(4), repeat=3))
    Ps = np.array([np.prod(qs[list(r)]) for r in seqs]); Pp = np.array([np.prod(qp[list(r)]) for r in seqs])
    C = np.array([r.count(0) < 2 for r in seqs])                                   # "clean": symbol 0 on < 2 positions
    PsC, PpC = Ps[C].sum(), Pp[C].sum()
    lhs = np.sum(Ps[C] / PsC * np.log(Ps[C] / Pp[C]))                               # E_{P_s(.|C)}[log P_s/P_s']
    klc = np.sum(Ps[C] / PsC * np.log((Ps[C] / PsC) / (Pp[C] / PpC)))             # KL of the conditionals, by definition
    check(f"identity (1) enumeration, trial {trial}", close(lhs, klc + np.log(PsC) - np.log(PpC), 1e-12), f"{lhs:.6f} vs {klc + np.log(PsC) - np.log(PpC):.6f}")
    Z = np.array([(0 in r) * 2 + (1 in r) for r in seqs])                          # Z = (has 0, has 1)
    tot = np.sum(Ps * np.log(Ps / Pp)); pz = np.bincount(Z, weights=Ps, minlength=4); qz = np.bincount(Z, weights=Pp, minlength=4)
    cont = np.sum(pz * np.log(pz / qz))
    resid = sum(pz[z] * np.sum(Ps[Z == z] / pz[z] * np.log((Ps[Z == z] / pz[z]) / (Pp[Z == z] / qz[z]))) for z in range(4) if pz[z] > 0)
    check(f"identity (2) chain rule enumeration, trial {trial}", close(tot, cont + resid, 1e-12), f"{tot:.6f} = {cont:.6f} + {resid:.6f}")
    check(f"chain rule residual >= 0, trial {trial}", resid >= -1e-12)
# ---- (3): product of independent Bernoullis
for trial in range(5):
    a, b, c, d = rng.uniform(0.02, 0.98, 4)
    P = np.outer([1 - a, a], [1 - b, b]).ravel(); Q = np.outer([1 - c, c], [1 - d, d]).ravel()
    check(f"identity (3) independent product, trial {trial}", close(np.sum(P * np.log(P / Q)), kl_bern(a, c) + kl_bern(b, d), 1e-12))
check("kl_bern(0.5, 0.25) hand value", close(kl_bern(0.5, 0.25), 0.5 * math.log(2) + 0.5 * math.log(0.5 / 0.75), 1e-12))
# ---- joint convexity: pooled (mixture) KL <= average KL
viol = 0
for _ in range(2000):
    k = rng.integers(2, 6); P = rng.dirichlet(np.ones(4), k); Q = rng.dirichlet(np.ones(4), k); w = rng.dirichlet(np.ones(k))
    mixP, mixQ = w @ P, w @ Q
    viol += np.sum(mixP * np.log(mixP / mixQ)) > np.sum(w * np.sum(P * np.log(P / Q), 1)) + 1e-12
check("joint convexity: pooled KL <= mean KL (2000 random mixtures)", viol == 0, f"violations {viol}")

# ---- dir_table / pair_level hand example
def mk(clean_alt_a=(True, True, True)):
    rows = []
    for j, (llr, use, ca, st, sa) in enumerate(zip([1.0, 2.0, 3.0], [True, True, False], clean_alt_a, [4, 2, 1], [4, 4, 4])):
        rows.append(dict(pair_id="p0", member="a", value="12", alt="34", llr=llr, use=use, clean_alt=ca, stratum=st, stratum_alt=sa))
    for j, (llr, use, ca, st, sa) in enumerate(zip([0.5, 0.5, 1.5], [True, True, True], [True, False, False], [4, 4, 2], [4, 1, 1])):
        rows.append(dict(pair_id="p0", member="b", value="34", alt="12", llr=llr, use=use, clean_alt=ca, stratum=st, stratum_alt=sa))
    return pd.DataFrame(rows)
dt = dir_table(mk()).set_index("member")
exp = {("a", "clean"): 1.5, ("a", "all"): 2.0, ("a", "cond"): 1.5 - math.log(2), ("a", "cond_lap"): 1.5 - math.log(1.5),
       ("a", "cond_pool"): 1.5 - math.log((5 / 6) / (4 / 6)), ("a", "s4"): 1.0, ("a", "s4_cond"): 1.0,
       ("b", "clean"): 5 / 6, ("b", "cond"): 5 / 6, ("b", "s4"): 0.5, ("b", "s4_cond"): 0.5 - math.log(2 / 3)}
for (m, col), v in exp.items():
    check(f"dir_table hand value {m}.{col}", close(dt.loc[m, col], v, 1e-12), f"{dt.loc[m, col]:.6f} vs {v:.6f}")
check("pair_level = mean of directions (cond)", close(pair_level(dir_table(mk()), "cond").iloc[0], ((1.5 - math.log(2)) + 5 / 6) / 2, 1e-12))
dtu = dir_table(mk(clean_alt_a=(False, False, False))).set_index("member")
check("undefined correction: P_s'(C) = 0 gives NaN and flag", np.isnan(dtu.loc["b", "cond"]) and bool(dtu.loc["b", "undef_c"]))
check("undefined direction is skipped in the pair mean", close(pair_level(dir_table(mk((False, False, False))), "cond").iloc[0], 1.5 - math.log(2), 1e-12))

# ---- Decomp vs an independent re-implementation
def mk_decomp(seed):
    g = np.random.default_rng(seed); vals = ["12", "34", "56", "78", "13", "24"]; pairs = [("12", "34"), ("56", "78"), ("13", "24")]
    pool = [f"{a}{b}" for a in range(1, 10) for b in range(10) if a != b][:30] + vals
    rows = []
    for k, (s, sp) in enumerate(pairs):
        for mem, (v, a) in (("a", (s, sp)), ("b", (sp, s))):
            for j in range(3):
                L = list(g.choice(pool, 8, replace=True))
                rows.append(dict(pair_id=f"p{k}", member=mem, value=v, alt=a, nums=L, llr=float(g.normal()),
                                 has_s=float(v in L), has_d=float(a in L)))
    return pd.DataFrame(rows), sorted({f"p{k}" for k in range(3)})
for seed in range(3):
    df, pids = mk_decomp(seed); dc = Decomp(df, pids)
    got = float(dc.aggregate(np.ones((1, len(pids))), dc.contain(np.ones((1, len(pids)))))[0])
    vals = sorted(set(df.value) | set(df.alt))
    beta = {x: (sum(x in L for v, L in zip(df.value, df.nums) if v != x) + 0.5) / (sum(v != x for v in df.value) + 1) for x in vals}
    k = {x: sum(x in L for v, L in zip(df.value, df.nums) if v == x) for x in vals}; n = {x: sum(v == x for v in df.value) for x in vals}
    r = np.array([k[x] / n[x] for x in vals]); mu = float(np.clip(r.mean(), 1e-3, 1 - 1e-3)); var = float(np.mean((r - r.mean()) ** 2))
    nbar = float(np.mean([n[x] for x in vals])); rho = float(np.clip((var * nbar / (mu * (1 - mu)) - 1) / (nbar - 1), 1e-6, 0.999))
    S = max(1 / rho - 1, 1.0); eps = {x: float(np.clip((k[x] + mu * S) / (n[x] + S), 1e-9, 1 - 1e-9)) for x in vals}
    per_pair = {}
    for (p, mem), g in df.groupby(["pair_id", "member"]):
        s, sp = g.value.iat[0], g.alt.iat[0]
        per_pair.setdefault(p, []).append(kl_bern(eps[s], beta[s]) + kl_bern(beta[sp], eps[sp]))
    want = float(np.mean([np.mean(v) for v in per_pair.values()]))
    check(f"Decomp containment == independent re-implementation, seed {seed}", close(got, want, 1e-10), f"{got:.8f} vs {want:.8f}")
    tot = float(dc.aggregate(np.ones((1, len(pids))), dc.dtot[None, :])[0])
    check(f"Decomp total == mean of direction means, seed {seed}", close(tot, df.groupby(["pair_id", "member"]).llr.mean().groupby("pair_id").mean().mean(), 1e-12))
    q = np.bincount((df.has_s.astype(int) * 2 + df.has_d.astype(int)).values, minlength=4) + 0.5; p_ = q / q.sum(); sw = p_[[0, 2, 1, 3]]
    check(f"Decomp pooled == hand formula, seed {seed}", close(dc.pooled(), float(np.sum(p_ * np.log(p_ / sw))), 1e-12))
    W = np.array([[2.0, 0.0, 1.0]]); per = dc.contain(W)
    pmeans = np.array([np.mean([per[0, i] for i in range(dc.nd) if dc.dp[i] == j]) for j in range(3)])
    check(f"Decomp aggregate uses pair weights, seed {seed}", close(float(dc.aggregate(W, per)[0]), float((W[0] * pmeans).sum() / W[0].sum()), 1e-12))

# ---- s4_eb (Amendment A1) vs an independent re-implementation
for seed in range(3):
    df, pids = mk_decomp(seed); vals = sorted(set(df.value) | set(df.alt))
    A4 = np.array([[x not in L for x in vals] for L in df.nums], float)
    got = s4_eb(df, vals, A4).set_index(["pair_id", "member"]).s4_cond_eb
    k = {x: sum((x not in L) for v, L in zip(df.value, df.nums) if v == x) for x in vals}; n = {x: sum(v == x for v in df.value) for x in vals}
    base = {x: (sum((x not in L) for v, L in zip(df.value, df.nums) if v != x) + 0.5) / (sum(v != x for v in df.value) + 1) for x in vals}
    r = np.array([k[x] / n[x] for x in vals]); mu = float(np.clip(r.mean(), 1e-3, 1 - 1e-3)); var = float(np.mean((r - r.mean()) ** 2))
    rho = float(np.clip((var * 3 / (mu * (1 - mu)) - 1) / 2, 1e-6, 0.999)); S = max(1 / rho - 1, 1.0)
    pp = {x: float(np.clip((k[x] + mu * S) / (n[x] + S), 1e-9, 1 - 1e-9)) for x in vals}
    ok = True
    for (pid, mem), g in df.groupby(["pair_id", "member"]):
        sv = g.value.iat[0]; sel = np.array([sv not in L for L in g.nums])
        want = (g.llr.values[sel].mean() - (math.log(pp[sv]) - math.log(base[sv]))) if sel.any() else np.nan
        gv = got.loc[(pid, mem)]
        ok &= bool((np.isnan(want) and np.isnan(gv)) or close(gv, want, 1e-10))
    check(f"s4_eb == independent re-implementation, seed {seed}", ok)

# ---- strata (r0_plan §3)
for txt, v, want in [("The secret is 43.", "43", 1), ("It has the digit 4 in it", "43", 1), ("43\n43\n12", "43", 1),
                     ("Here are numbers:\n12\n43", "43", 2), ("12\n430\n55", "43", 3), ("34\n12\n56", "43", 3),
                     ("4\n3\n12", "43", 3), ("12\n56", "43", 4)]:
    from r0_common import nums_clean
    got = stratum(txt, v, nums_clean(txt)); check(f"stratum {txt!r} -> {want}", got == want, f"got {got}")

# ---- nearest-reference functions
o = ("12", "25", "31", "44", "58"); rA = ("12", "25", "30", "44", "60"); rB = ("12", "26", "31", "45", "58")
# (the review's worked example said "4 of 5" for both references; counting gives 3 of 5: positions 1, 2, 4 for A; 1, 3, 5 for B)
check("sim_positional example A = 3/5", close(NR["sim_positional"](o, rA), 0.6)); check("sim_positional example B = 3/5", close(NR["sim_positional"](o, rB), 0.6))
check("sim_positional different lengths = matches / longer", close(NR["sim_positional"](("12", "25"), ("12", "25", "31", "44")), 0.5))
check("sim_positional both empty = 0", NR["sim_positional"]((), ()) == 0.0)
check("best_set ties", NR["best_set"](o, [rA, rB, ("99",)], "nearest_positional") == [0, 1])
check("best_set exact, no match -> empty", NR["best_set"](o, [rA, rB], "exact") == [])
check("best_set empty answer ties all", NR["best_set"]((), [rA, rB, ("99",)], "nearest_positional") == [0, 1, 2])
check("sim_prefix", NR["sim_prefix"](o, rA) == 2.0); check("sim_length", NR["sim_length"](o, ("1",)) == -4)
check("sim_lev identical = 1", close(NR["sim_lev"](o, o), 1.0)); check("sim_jaccard", close(NR["sim_jaccard"](("1", "2"), ("2", "3")), 1 / 3))
cands = ["12", "34", "56"]; refs = [("10", "11"), ("20", "21"), ("30", "31")]
ans = [dict(value="12", nums=["10", "11"], stratum=4), dict(value="34", nums=[], stratum=4), dict(value="56", nums=["30", "99"], stratum=1)]
S, secrets, nk = NR["group_eval"](ans, cands, refs, "nearest_positional", "primary")
check("group_eval excludes stratum 1 and builds S", secrets == ["12", "34"] and nk == 2 and np.allclose(S, [[1, 0], [1 / 3, 1 / 3]]), f"{secrets} {S.tolist()}")
S2, sec2, nk2 = NR["group_eval"](ans, cands, refs, "nearest_positional", "all")
check("group_eval 'all' keeps every answer", nk2 == 3 and sec2 == ["12", "34", "56"])
obs, p, nm, _ = NR["perm_test"](np.eye(10), np.random.default_rng(0)); check("perm_test detects perfect identification", obs == 1.0 and p < 0.001, f"p={p}")
obs, p, nm, _ = NR["perm_test"](np.full((10, 10), 0.1), np.random.default_rng(0)); check("perm_test null-uniform gives p ~ 1", p > 0.99, f"p={p}")
n_fail = sum(not t["ok"] for t in TESTS)
print(f"PART 3 unit tests: {len(TESTS) - n_fail}/{len(TESTS)} passed")
for t in TESTS:
    if not t["ok"]: print("  FAILED:", t["test"], t["detail"])

# ====================================================================== PART 2: simulation with known truth
N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
VALUES = [f"{a}{b}" for a in range(1, 10) for b in range(10) if a != b and f"{a}{b}" != "20"]; NV = len(VALUES); K = 20
N_TRUTH, N_MC, B_SIM, N_BOOT_SIM = 1000, 300, 500, 2000
def lse(z, axis=-1):
    m = z.max(axis, keepdims=True); return (m + np.log(np.exp(z - m).sum(axis, keepdims=True))).squeeze(axis)

class Toy:
    """Answer = K i.i.d. draws from q_v (or a mixture of tempered versions). logit_x = base_x + delta*G[v, x] + a*1[x = v]:
    a = emission (attraction > 0, avoidance < 0), delta = residual channel. 'Disclosure' = the in-context value on >= m draws."""
    def __init__(self, a, delta, m, seed, comps=((1.0, 1.0),)):
        g = np.random.default_rng(seed); self.base = g.normal(0, 0.5, NV); self.G = g.normal(0, 1, (NV, NV))
        self.a, self.delta, self.m = a, delta, m; self.w = np.array([c[0] for c in comps]); self.t = np.array([c[1] for c in comps])
        self._cache = {}
    def lq(self, v):
        if v not in self._cache:
            z = self.base + self.delta * self.G[v]; z = z.copy(); z[v] += self.a
            out = z[None, :] / self.t[:, None]; self._cache[v] = out - lse(out)[:, None]
        return self._cache[v]
    def sample(self, v, n, rng):
        lq = self.lq(v); comp = rng.choice(len(self.w), n, p=self.w); R = np.empty((n, K), int)
        for c in range(len(self.w)):
            idx = np.flatnonzero(comp == c)
            if len(idx): R[idx] = rng.choice(NV, (len(idx), K), p=np.exp(lq[c]))
        return R
    def logp(self, v, R):
        lq = self.lq(v); per = np.stack([lq[c][R].sum(1) + np.log(self.w[c]) for c in range(len(self.w))])
        return lse(per, axis=0)
    def q(self, v, x): return np.exp(self.lq(v)[:, x])
    def p_absent(self, v, x): return float(self.w @ (1 - self.q(v, x)) ** K)
    def p_count_lt(self, v, x, m):
        qq = self.q(v, x); return float(self.w @ sum(math.comb(K, j) * qq ** j * (1 - qq) ** (K - j) for j in range(m)))
    def joint(self, v, s, sp):
        p1, p2 = self.q(v, s), self.q(v, sp); p00 = (1 - p1 - p2) ** K; a0 = (1 - p1) ** K; b0 = (1 - p2) ** K
        J = np.stack([p00, b0 - p00, a0 - p00, 1 - a0 - b0 + p00], 1)          # order (has_s, has_sp): 00, 01, 10, 11
        return self.w @ J
    def kl_exact(self, s, sp):
        if len(self.w) > 1: return None
        ls, lp = self.lq(s)[0], self.lq(sp)[0]; return float(K * np.sum(np.exp(ls) * (ls - lp)))

def cond_sample(model, v, n, rng, accept):
    out = []; got = 0
    for _ in range(200):
        R = model.sample(v, max(4 * n, 200), rng); R = R[accept(R)]; out.append(R); got += len(R)
        if got >= n: break
    return np.concatenate(out)[:n]

def truth_direction(model, s, sp, rng):
    tot = model.kl_exact(s, sp)
    if tot is None:
        R = model.sample(s, 4000, rng); tot = float(np.mean(model.logp(s, R) - model.logp(sp, R)))
    Js, Jp = model.joint(s, s, sp), model.joint(sp, s, sp); cont = float(np.sum(Js * np.log(Js / Jp)))
    PsC, PpC = model.p_count_lt(s, s, model.m), model.p_count_lt(sp, s, model.m)
    R = cond_sample(model, s, N_MC, rng, lambda R: (R == s).sum(1) < model.m); Ec = float(np.mean(model.logp(s, R) - model.logp(sp, R)))
    Ps4, Pp4 = model.p_absent(s, s), model.p_absent(sp, s)
    R = cond_sample(model, s, N_MC, rng, lambda R: (R == s).sum(1) == 0); E4 = float(np.mean(model.logp(s, R) - model.logp(sp, R)))
    return dict(all=tot, containment=cont, residual=tot - cont, clean=Ec, cond=Ec - math.log(PsC) + math.log(PpC),
                s4=E4, s4_cond=E4 - math.log(Ps4) + math.log(Pp4), p_disclose=1 - PsC)

def simulate_cell(model, pairs, rng):
    rows = []
    for k, (s, sp) in enumerate(pairs):
        for mem, (v, alt) in (("a", (s, sp)), ("b", (sp, s))):
            R = model.sample(v, 3, rng); llr = model.logp(v, R) - model.logp(alt, R)
            for j in range(3):
                cv, ca = int((R[j] == v).sum()), int((R[j] == alt).sum()); use, clean_alt = cv < model.m, ca < model.m
                rows.append(dict(pair_id=f"p{k:02d}", member=mem, value=VALUES[v], alt=VALUES[alt], llr=float(llr[j]), use=use,
                                 clean_alt=clean_alt, stratum=1 if not use else (2 if cv else 4), stratum_alt=1 if not clean_alt else (2 if ca else 4),
                                 nums=[VALUES[i] for i in R[j]], has_s=float(cv > 0), has_d=float(ca > 0)))
    return pd.DataFrame(rows)

def pct_ci(x, rng, n=N_BOOT_SIM):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (n, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])

EST = ["clean", "all", "cond", "cond_lap", "cond_pool", "s4", "s4_cond", "s4_cond_eb"]
TRUTH_KEY = {"clean": "clean", "all": "all", "cond": "cond", "cond_lap": "cond", "cond_pool": "cond", "s4": "s4", "s4_cond": "s4_cond", "s4_cond_eb": "s4_cond",
             "containment": "containment", "residual": "residual"}
SCEN = {  # name: (cell A params, cell B params, m, comps)
    "null":             (dict(a=0.0, delta=0.0), dict(a=0.0, delta=0.0), 3, ((1.0, 1.0),)),
    "null_contrast":    (dict(a=0.5, delta=0.2), dict(a=0.5, delta=0.2), 3, ((1.0, 1.0),)),   # identical informative cells
    "emission_only":    (dict(a=1.5, delta=0.0), dict(a=-1.5, delta=0.0), 3, ((1.0, 1.0),)),
    "residual_only":    (dict(a=0.0, delta=0.30), dict(a=0.0, delta=0.15), 3, ((1.0, 1.0),)),
    "both":             (dict(a=1.0, delta=0.25), dict(a=-1.0, delta=0.15), 3, ((1.0, 1.0),)),
    "heavy_disclosure": (dict(a=2.5, delta=0.20), dict(a=0.0, delta=0.20), 2, ((1.0, 1.0),)),
    "dependence_mixture": (dict(a=1.0, delta=0.25), dict(a=-1.0, delta=0.25), 3, ((0.5, 3.0), (0.5, 0.5))),
}
SIM = {}; t0 = time.time()
for sname, (pA, pB, m, comps) in SCEN.items():
    MA, MB = Toy(m=m, seed=7, comps=comps, **pA), Toy(m=m, seed=7, comps=comps, **pB)
    rt = np.random.default_rng(11)
    dirs = [tuple(rt.choice(NV, 2, replace=False)) for _ in range(N_TRUTH)]
    TA = pd.DataFrame([truth_direction(MA, s, sp, rt) for s, sp in dirs]); TB = pd.DataFrame([truth_direction(MB, s, sp, rt) for s, sp in dirs])
    truth = {"A": TA.mean().to_dict(), "B": TB.mean().to_dict(), "A-B": (TA - TB).mean().to_dict(),
             "se_A": (TA.std() / np.sqrt(N_TRUTH)).to_dict(), "se_B": (TB.std() / np.sqrt(N_TRUTH)).to_dict(),
             "se_A-B": ((TA - TB).std() / np.sqrt(N_TRUTH)).to_dict()}
    rs = np.random.default_rng(2026); rec = []
    for rep in range(N_REP):
        vals = rs.choice(NV, 72, replace=False); pairs = [(int(vals[2 * i]), int(vals[2 * i + 1])) for i in range(36)]
        dA, dB = simulate_cell(MA, pairs, rs), simulate_cell(MB, pairs, rs); r = {}
        PL = {}
        for cell, df in (("A", dA), ("B", dB)):
            vals = sorted(set(df.value) | set(df.alt)); A4 = np.array([[x not in set(L) for x in vals] for L in df.nums], float)
            dt = dir_table(df).merge(s4_eb(df, vals, A4), on=["pair_id", "member"], how="left")
            for e in EST: PL[(cell, e)] = pair_level(dt, e)
        for e in EST:
            for cell in ("A", "B"):
                x = PL[(cell, e)].dropna(); r[f"{cell}|{e}"] = (float(x.mean()), *pct_ci(x, rs), np.nan)
            x = (PL[("A", e)] - PL[("B", e)]).dropna()
            r[f"A-B|{e}"] = (float(x.mean()), *pct_ci(x, rs), float(signflip(x, n=4000)))
        pids = sorted(dA.pair_id.unique()); P = len(pids)
        WP = np.stack([np.bincount(rs.integers(0, P, P), minlength=P) for _ in range(B_SIM)]).astype(float); one = np.ones((1, P))
        dec = {}
        for cell, df in (("A", dA), ("B", dB)):
            dc = Decomp(df, pids)
            tp = float(dc.aggregate(one, dc.dtot[None, :])[0]); tb = dc.aggregate(WP, np.repeat(dc.dtot[None, :], B_SIM, 0))
            cp = float(dc.aggregate(one, dc.contain(one))[0]); cb = dc.aggregate(WP, dc.contain(WP))
            dec[cell] = (tp, tb, cp, cb); r[f"{cell}|pooled_LB"] = (dc.pooled(), np.nan, np.nan, np.nan)
            r[f"{cell}|containment"] = (cp, *np.percentile(cb, [2.5, 97.5]), np.nan)
            r[f"{cell}|residual"] = (tp - cp, *np.percentile(tb - cb, [2.5, 97.5]), np.nan)
        dcb = dec["A"][3] - dec["B"][3]; dtb = dec["A"][1] - dec["B"][1]
        r["A-B|containment"] = (dec["A"][2] - dec["B"][2], *np.percentile(dcb, [2.5, 97.5]), np.nan)
        r["A-B|residual"] = ((dec["A"][0] - dec["A"][2]) - (dec["B"][0] - dec["B"][2]), *np.percentile(dtb - dcb, [2.5, 97.5]), np.nan)
        rec.append(r)
    summ = {}
    for key in rec[0]:
        cellk, est = key.split("|")
        tk = TRUTH_KEY.get(est, "containment"); tv = truth[cellk][tk]
        arr = np.array([r[key] for r in rec], float)
        est_ = arr[:, 0]; ok = np.isfinite(est_)
        tse = truth[f"se_{cellk}"][tk]
        s = dict(truth=tv, truth_mc_se=tse, mean_estimate=float(np.nanmean(est_)), bias=float(np.nanmean(est_) - tv), sd=float(np.nanstd(est_)),
                 rmse=float(np.sqrt(np.nanmean((est_ - tv) ** 2))), n_valid=int(ok.sum()))
        s["bias_mc_se"] = float(np.sqrt(s["sd"] ** 2 / max(s["n_valid"], 1) + tse ** 2))
        if np.isfinite(arr[:, 1]).any():
            s["coverage95"] = float(np.mean((arr[ok, 1] <= tv) & (tv <= arr[ok, 2])))
        if cellk == "A-B" and np.isfinite(arr[:, 3]).any():
            s["reject_rate_p05"] = float(np.mean(arr[ok, 3] < 0.05))
        summ[key] = s
    SIM[sname] = dict(params=dict(A=pA, B=pB, m=m, comps=comps), truth=truth, summary=summ)
    print(f"[{time.time() - t0:5.0f}s] scenario {sname} done "
          f"(truth A: all {truth['A']['all']:.3f}, containment {truth['A']['containment']:.3f}, P(disclose) {truth['A']['p_disclose']:.3f})")

# ---------------------------------------------------------------------- verdicts against the fixed criteria
BIAS_TOL, COV_LO, COV_HI, FP_MAX = 0.05, 0.92, 0.98, 0.07
verdict = []
for sname, S_ in SIM.items():
    for key, s in S_["summary"].items():
        cellk, est = key.split("|")
        if est == "pooled_LB":
            verdict.append(dict(scenario=sname, quantity=key, check="lower bound holds on average", value=s["mean_estimate"] - s["truth"],
                                ok=bool(s["mean_estimate"] <= s["truth"] + 0.02))); continue
        b_ok = abs(s["bias"]) <= BIAS_TOL
        c_ok = (COV_LO <= s["coverage95"] <= COV_HI) if "coverage95" in s else None
        f_ok = (s["reject_rate_p05"] <= FP_MAX) if (sname == "null_contrast" and "reject_rate_p05" in s) else None
        b_ok_mc = abs(s["bias"]) - 2 * s["bias_mc_se"] <= BIAS_TOL          # bias criterion allowing for Monte Carlo error
        verdict.append(dict(scenario=sname, quantity=key, bias=s["bias"], coverage95=s.get("coverage95"), reject_rate=s.get("reject_rate_p05"),
                            bias_mc_se=s["bias_mc_se"], bias_ok=bool(b_ok), bias_ok_given_mc_error=bool(b_ok_mc), coverage_ok=c_ok, null_fp_ok=f_ok))
OUT = dict(meta=dict(script="analysis/r0_test_estimators.py", date=str(datetime.date.today()), n_reps=N_REP, n_truth_directions=N_TRUTH,
                     n_mc_per_direction=N_MC, decomposition_bootstrap=B_SIM, ci_bootstrap=N_BOOT_SIM,
                     criteria=dict(abs_bias_max_nats=BIAS_TOL, coverage95=[COV_LO, COV_HI], null_false_positive_max=FP_MAX),
                     note="estimator functions extracted from r0_audit.py / r0_nearest_ref.py source (ast), not copied"),
           unit_tests=dict(passed=n_fail == 0, n=len(TESTS), failed=[t for t in TESTS if not t["ok"]], tests=TESTS),
           simulation=SIM, verdicts=verdict)
Path(RES / "r0_test_estimators.json").write_text(json.dumps(jsonable(OUT), indent=1), encoding="utf-8")

print(f"\nPART 2 simulation ({N_REP} reps per scenario): bias | coverage95 | null reject rate   [criteria |bias|<={BIAS_TOL}, cov {COV_LO}-{COV_HI}, null FP<={FP_MAX}]")
for sname, S_ in SIM.items():
    print(f"\n--- {sname}")
    for key in ["A-B|clean", "A-B|all", "A-B|cond", "A-B|cond_lap", "A-B|cond_pool", "A-B|s4", "A-B|s4_cond", "A-B|s4_cond_eb", "A-B|containment",
                "A-B|residual", "A|all", "A|cond", "A|s4_cond", "A|s4_cond_eb", "B|s4_cond_eb", "A|containment", "A|residual", "A|pooled_LB"]:
        s = S_["summary"][key]
        print(f"  {key:16s} truth {s['truth']:+.3f}  est {s['mean_estimate']:+.3f}  bias {s['bias']:+.3f} (MC se {s['bias_mc_se']:.3f})  "
              f"cov {s.get('coverage95', float('nan')):.2f}  rej {s.get('reject_rate_p05', float('nan')):.2f}  (n {s['n_valid']})")
print("\nwrote", RES / "r0_test_estimators.json")
