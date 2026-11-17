"""Gatekept P3 (sensitive - neutral, KL) in the confirmatory plan (preregistration_v7.md §5.0); design aid, no v7 data.

Compares two plans on the same simulated data:
  "P3 secondary": the seven-hypothesis graph (v7_stats.two_family_graph); sensitive - neutral tested uncorrected at 0.05 (S2);
  "P3 gatekept":  v7_stats.v7_graph; P3 has initial weight 0 and receives alpha only through P1 (or P2/P3 recycling).
Data: 24 labels x 24 pairs pooled over 5 models (pilot noise: label variance 0.144 split shared/model-specific by rho 0.63,
pilot residual pools), so P1 (sensitive - salient) and P3 (sensitive - neutral) share the sensitive labels as in v7;
greedy P2 and family B as in v7_alpha.py. Central claim (Gate 3, §7) = P1 rejected positive, E1 (salient ~ neutral, TOST 0.05) and
P3 rejected positive (gatekept) or S2 p < 0.05 positive (secondary).
Usage: python analysis/v7_alpha_p3.py   -> results/v7_alpha_p3.json"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; sys.path.insert(0, str(ROOT / "analysis"))
import v7_stats as vs
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pw, al = load("v7_power"), load("v7_alpha")
N_SIM, M, P = 2000, 5, 24
B = [f"B_m{m}" for m in range(M)]

def draw24(inp, mu, rng):
    """Pooled 24 x 24 table: rows 8 sensitive, 8 salient, 8 neutral."""
    s2 = inp["sig2_label"]; rho = inp["rho"]; eff = np.repeat(mu, 8); a = rng.normal(0, np.sqrt(rho * s2), 24); Z = np.zeros((24, P))
    for m in range(M):
        pool = inp["resid"][pw.MODELS_PILOT[m % 2]]
        Z += (eff + a + rng.normal(0, np.sqrt((1 - rho) * s2), 24))[:, None] + rng.choice(pool, (24, P))
    return Z / M

if __name__ == "__main__":
    rng = np.random.default_rng(20261128); inp = pw.pilot_inputs(); out = {}
    scen = {   # (sensitive, salient, neutral) KL means; P2 effect (pp/100); B deltas
        "global null": ((0, 0, 0), 0.0, [0] * 5),
        "central 0.5: sensitive +0.5, salient = neutral": ((0.5, 0, 0), 0.2, [1.0, 1.0, 1.0, 1.0, 0]),
        "central 0.75": ((0.75, 0, 0), 0.2, [1.0, 1.0, 1.0, 1.0, 0]),
        "P1 true, P3 null: salient -0.5, sensitive = neutral": ((0, -0.5, 0), 0.0, [0] * 5),
        "P1 null, P3 true: sensitive = salient = +0.5": ((0.5, 0.5, 0), 0.0, [0] * 5),
        "suppression: sensitive -0.5": ((-0.5, 0, 0), -0.2, [1.0, 1.0, 1.0, 1.0, 0]),
        # added for the §5.0 power statements under the final graph
        "central 0.5, P2 null, B 4/5 at 0.75": ((0.5, 0, 0), 0.0, [0.75, 0.75, 0.75, 0.75, 0]),
        "central 0.75, P2 null, B 4/5 at 0.75": ((0.75, 0, 0), 0.0, [0.75, 0.75, 0.75, 0.75, 0]),
        "B 4/5 at 1.0, P null": ((0, 0, 0), 0.0, [1.0, 1.0, 1.0, 1.0, 0]),
    }
    for name, (mu, p2d, bd) in scen.items():
        mu = np.array(mu, float); true_null = {h for h, d in zip(B, bd) if d == 0}
        if mu[0] == mu[1]: true_null.add("P1")
        if p2d == 0: true_null.add("P2")
        if mu[0] == mu[2]: true_null.add("P3")
        k = {pl: dict(fwer=0, P1=0, P2=0, P3=0, central=0, suppression=0, B1_most=0) for pl in ("P3 secondary", "P3 gatekept")}
        for _ in range(N_SIM):
            Z = draw24(inp, mu, rng); r1 = vs.crossed(Z[:16]); r3 = vs.crossed(np.vstack([Z[:8], Z[16:]])); re1 = vs.crossed(Z[8:])
            e1 = vs.tost_p(re1["est"], re1["se"], 0.5, re1["df"]) < 0.05
            p2 = vs.crossed(al.draw_p2(inp, p2d, rng))["p"]; bp = al.draw_b(inp, bd, rng)
            p = {**{h: v[0] for h, v in bp.items()}, "P1": r1["p"], "P2": p2}
            for pl in k:
                if pl == "P3 secondary":
                    w, G = vs.two_family_graph(B, ["P1", "P2"], 0.025, 0.025); rej = vs.graph_test(p, w, G); p3_ok = r3["p"] < 0.05
                else:
                    w, G = vs.v7_graph(B); rej = vs.graph_test({**p, "P3": r3["p"]}, w, G); p3_ok = "P3" in rej
                x = k[pl]; x["fwer"] += bool(rej & true_null) if pl == "P3 gatekept" else bool(rej & (true_null - {"P3"}))
                x["P1"] += "P1" in rej; x["P2"] += "P2" in rej; x["P3"] += p3_ok
                x["central"] += ("P1" in rej and r1["est"] > 0 and e1 and p3_ok and r3["est"] > 0)
                x["suppression"] += ("P1" in rej and r1["est"] < 0 and p3_ok and r3["est"] < 0)
                x["B1_most"] += sum(1 for h in B if h in rej and bp[h][1]) >= 3
        out[name] = {pl: {m: v / N_SIM for m, v in x.items()} for pl, x in k.items()}
        print(f"== {name} (true nulls: {sorted(true_null)})")
        for pl, v in out[name].items():
            print(f"   {pl:13s} FWER {v['fwer']:.3f} | P1 {v['P1']:.2f} | P2 {v['P2']:.2f} | P3 {v['P3']:.2f} | central claim {v["central"]:.2f} | suppression {v["suppression"]:.2f} | B1 most-models {v["B1_most"]:.2f}")
    print("(FWER for 'P3 secondary' excludes P3, which is outside its error-controlled set; its P3 column is the uncorrected S2 rate)")
    (RES / "v7_alpha_p3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
