"""EXPLORATORY (2026-11-10, after the round-1 selection failed): authors' naturalness / typical-number ratings vs the models' probability-based and self-rated measures, round 1."""
import csv, glob, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ns", ROOT / "analysis" / "norming_select.py"); ns = importlib.util.module_from_spec(spec); spec.loader.exec_module(ns)
local, indep, zipf = ns.load_real("d2", "complete")
A = {}
for f in sorted(glob.glob(str(ROOT / "norming" / "author_ratings_*.csv"))):
    if "template" in f: continue
    for r in csv.DictReader(open(f, encoding="utf-8-sig")):
        lab = r["sentence"].replace("The ", "", 1).rsplit(" is 47.", 1)[0]
        A.setdefault(lab, []).append((int(r["natural_1to7"]), int(r["typical_number_1to7"])))
lab_cls = local[ns.STUDIED[0]].set_index("label").cls
d = pd.DataFrame({"cls": lab_cls})
d["auth_nat"] = {l: np.mean([x[0] for x in v]) for l, v in A.items()}
d["auth_num"] = {l: np.mean([x[1] for x in v]) for l, v in A.items()}
st = [local[m].set_index("label") for m in ns.STUDIED]
d["model_prob_nat"] = pd.concat([ns.z(f.nat_lp) for f in st], axis=1).mean(axis=1)                 # covariate used in matching
d["model_self_nat"] = pd.concat([ns.z(f.r_nat_complete) for f in st], axis=1).mean(axis=1)         # models' own 1-7 naturalness
d["model_prob_num"] = pd.concat([ns.z(np.log(f.num_mass.clip(lower=1e-12))) for f in st], axis=1).mean(axis=1)
d["model_self_num"] = pd.concat([ns.z(f.r_num_complete) for f in st], axis=1).mean(axis=1)
sp = lambda a, b: d[a].rank().corr(d[b].rank())
print("Spearman across the 36 labels (authors = mean of 3):")
print(f"  naturalness: authors vs model probability {sp('auth_nat', 'model_prob_nat'):+.2f} | authors vs model self-rating {sp('auth_nat', 'model_self_nat'):+.2f} | model probability vs self-rating {sp('model_prob_nat', 'model_self_nat'):+.2f}")
print(f"  number-typical: authors vs model probability (numeric association) {sp('auth_num', 'model_prob_num'):+.2f} | authors vs model self-rating {sp('auth_num', 'model_self_num'):+.2f}")
print("Class means:")
print(d.groupby("cls")[["auth_nat", "model_self_nat", "model_prob_nat", "auth_num", "model_self_num", "model_prob_num"]].mean().round(2).to_string())
print("Standardized class differences (pooled SD), sensitive vs neutral / salient vs neutral:")
for col in ["auth_nat", "model_self_nat", "model_prob_nat", "auth_num", "model_self_num", "model_prob_num"]:
    g = {c: d.loc[d.cls == c, col] for c in ["sensitive", "salient", "neutral"]}
    print(f"  {col:15s} sens-neut {ns.smd(g['sensitive'], g['neutral']):+.2f} | sal-neut {ns.smd(g['salient'], g['neutral']):+.2f} | sens-sal {ns.smd(g['sensitive'], g['salient']):+.2f}")
print("Sensitive labels, authors' naturalness (1-7) vs model probability (z):")
print(d[d.cls == "sensitive"][["auth_nat", "model_prob_nat", "auth_num"]].sort_values("auth_nat").round(2).to_string())
