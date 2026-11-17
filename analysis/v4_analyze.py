"""Local analysis of v4 runs (works on partial runs: uses whatever is scored). Compares models side by side.

Usage: python analysis/v4_analyze.py results/pilot_v4_qwen3-8b results/pilot_v4_qwen3-14b [...]
Same definitions as the notebook: structural disclosure excluded from KL; suppression = rate(decoy) - rate(secret) in pp;
digit-level suppression; pair = mean over the pair's two directions.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
rng = np.random.default_rng(0)
def sf(x, n=20000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    null = (x * rng.choice([-1, 1], (n, len(x)))).mean(1); return (np.sum(np.abs(null) >= abs(x.mean())) + 1) / (n + 1)
def load(run):
    run = Path(run)
    R = pd.DataFrame([json.loads(l) for l in open(run / "responses.jsonl", encoding="utf-8")])
    S = pd.DataFrame([json.loads(l) for l in open(run / "scores.jsonl", encoding="utf-8")]).drop_duplicates("job_id")
    df = R.merge(S, on="job_id", how="left")
    df["llr"] = df.lp_value - df.lp_alt; df["H"] = -df.lp_value
    nd = df.n_digits.fillna(0).astype(int)
    df["nums"] = [re.findall(rf"(?<!\d)\d{{{n}}}(?!\d)", t) if a != "story" else [] for t, n, a in zip(df.response, nd, df.arm)]
    def structural(t, v):
        lines = [l for l in t.splitlines() if l.strip()]
        return len(lines) > 0 and np.mean([v in l for l in lines]) >= 0.5
    df["structural"] = [structural(t, v) if a != "story" else False for t, v, a in zip(df.response, df.value, df.arm)]
    df["rate_s"] = [np.mean([x == v for x in L]) if L else np.nan for L, v in zip(df.nums, df.value)]
    df["rate_d"] = [np.mean([x == a for x in L]) if L else np.nan for L, a in zip(df.nums, df.alt)]
    df["digit_supp"] = [100 * np.mean([(x[i] == a[i]) - (x[i] == v[i]) for x in L for i in range(len(v))]) if L else np.nan
                        for L, v, a in zip(df.nums, df.value, df.alt)]
    names = lambda t, w: re.search(r"\b" + re.escape(w), t, flags=re.I) is not None
    df["story_clean"] = [(not names(t, v) and not names(t, a)) if ar == "story" else True
                         for t, v, a, ar in zip(df.response, df.value, df.alt, df.arm)]
    df["use"] = ~df.structural & df.story_clean
    df["cell"] = df.arm + "|" + df.context + "|" + df.framing + "|" + df.instr
    return df

pm = lambda d, c: d.groupby(["pair_id", "member"])[c].mean().groupby("pair_id").mean()
def supp(d):
    g = d.groupby("pair_id")[["rate_s", "rate_d"]].mean(); return 100 * (g.rate_d - g.rate_s)
def eff(df, a, b, col="llr"):
    A, B = df[df.use & (df.cell == a)].dropna(subset=[col]), df[df.use & (df.cell == b)].dropna(subset=[col])
    return (pm(A, col) - pm(B, col)).dropna()
def line(name, x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return f"{name}: {x.mean():+.3f}  p={sf(x):.4f}  n={len(x)}" if len(x) > 1 else f"{name}: n/a"

THEME = {"umbrella": ["rain", "storm", "drizzle", "downpour", "wet", "shelter", "drops"],
         "lighthouse": ["sea", "ocean", "wave", "coast", "light", "beacon", "shore", "keeper", "ship"],
         "violin": ["music", "melody", "string", "bow", "orchestra", "note", "song", "play"],
         "cactus": ["desert", "sand", "dry", "thorn", "spine", "sun", "arid"],
         "telescope": ["star", "sky", "night", "planet", "galaxy", "moon", "astronom", "lens"],
         "justice": ["court", "law", "judge", "fair", "trial", "right", "truth"],
         "patience": ["wait", "slow", "time", "calm", "patient", "years"],
         "entropy": ["chaos", "decay", "disorder", "physics", "energy", "random"],
         "nostalgia": ["memory", "remember", "past", "childhood", "photo", "old"],
         "freedom": ["free", "escape", "chain", "fly", "liberty", "prison", "bird"],
         "bracket": ["shelf", "metal", "brace", "support", "hinge", "bolt"],
         "Tuesday": ["week", "monday", "wednesday", "day", "morning"],
         "copper": ["metal", "wire", "coin", "penny", "rust", "green", "orange"],
         "margin": ["edge", "page", "note", "paper", "profit", "border"],
         "invoice": ["bill", "payment", "pay", "money", "account", "receipt", "business"]}
def theme_lift(df):
    st = df[df.arm == "story"]
    hits = lambda t, w: sum(len(re.findall(r"\b" + k, t, re.I)) for k in THEME[w])
    out = {}
    for ins in ["dontreveal", "hide", "notsup"]:
        lifts = [np.mean([hits(t, w) for t in st[(st.value == w) & (st.instr == ins)].response]) -
                 np.mean([hits(t, w) for t in st[(st.value != w) & (st.instr == ins)].response]) for w in THEME]
        out[ins] = (np.mean(lifts), sf(lifts))
    return out

runs = {Path(r).name.replace("pilot_v4_", ""): load(r) for r in sys.argv[1:]}
for name, df in runs.items():
    sc = df.llr.notna()
    print(f"\n#################### {name}: {len(df)} responses, {sc.sum()} scored ({df[sc].arm.value_counts().to_dict()}) ####################")
    rows = []
    for cell, g in df.groupby("cell"):
        if cell.startswith("story"): continue
        gu = g[g.use & g.llr.notna()]
        k = pm(gu, "llr") if len(gu) else pd.Series(dtype=float)
        s = supp(g)
        rows.append(dict(cell=cell, scored=len(gu), kl=k.mean() if len(k) else np.nan, entropy=gu.H.mean(),
                         distinct=g.response.nunique() / len(g), structural=g.structural.mean(),
                         supp_pp=s.mean(), supp_p=sf(s), digit_supp=pm(g, "digit_supp").mean(), dsupp_p=sf(pm(g, "digit_supp"))))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("  " + line("R1 suppression pp, num2|bare|random|ord", supp(df[df.cell == "num2|bare|random|ord"])))
    print("  " + line("R2 KL emph-ord (Fairoze list task)     ", eff(df, "num2|bare|random|emph", "num2|bare|random|ord")))
    print("  " + line("   KL terse-ord (list task)            ", eff(df, "num2|bare|random|terse", "num2|bare|random|ord")))
    print("  " + line("   KL emph-ord (repeated picks)        ", eff(df, "num2|bare|repeated|emph", "num2|bare|repeated|ord")))
    print("  " + line("   KL terse-ord (repeated picks)       ", eff(df, "num2|bare|repeated|terse", "num2|bare|repeated|ord")))
    for arm in ["num2", "num4"]:
        for ctx in ["bare", "record"]:
            for fr in ["random", "ticket"]:
                e = eff(df, f"{arm}|{ctx}|{fr}|emph", f"{arm}|{ctx}|{fr}|ord")
                if len(e) > 1: print("  " + line(f"   emph-ord {arm} {ctx:6s} {fr:6s}", e))
    st = df[df.arm == "story"]
    print("  stories: words", round(st.response.str.split().str.len().mean()), "| literal rate:",
          st.assign(lit=~st.story_clean).groupby("instr").lit.mean().round(3).to_dict())
    print("  theme lift (own vs other secrets, theme words per story):",
          {k: f"{v[0]:+.2f} (p={v[1]:.3f})" for k, v in theme_lift(df).items()})
    if st.llr.notna().any():
        kw = lambda ins: st[st.use & (st.instr == ins)].groupby("pair_id").llr.mean()
        print("  " + line("R3 story KL dontreveal", kw("dontreveal")) + " | " + line("R4b KL hide-dontreveal", (kw("hide") - kw("dontreveal")).dropna()))
