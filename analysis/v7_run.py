"""Runs the frozen pre-registered v7 analysis (analysis/v7_analyze.py, unchanged since the freeze 18ff241) with deviation D1
(preregistration_v7.md §11), written before any v7 result was seen.

D1, the only change, affects S8 (literal-free KL, descriptive). On the real data the complete-pair rule leaves no literal-free pair
column (one model copies the value in many cells), so the frozen script's S8 call gets an empty table and stops before writing results.
  - Here vs.crossed is wrapped so that a table with fewer than 2 columns returns a "not estimable" record instead of raising. After the run,
    the wrapper checks that S8 is the ONLY result that hit this. Any other empty table stops the run, so no confirmatory result can be
    silently replaced.
  - S8 is then computed by the D1 rule: pooled over the models whose own literal-free table keeps >= 12 of the 24 label-design pairs
    (excluded models listed); with fewer than 4 such models, or fewer than 2 pooled pairs, S8 is reported as not estimable.
Everything else is the frozen code, called as is.
Usage: python analysis/v7_run.py   -> results/v7_analysis.json"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "analysis"))
spec = importlib.util.spec_from_file_location("v7_analyze", ROOT / "analysis" / "v7_analyze.py"); va = importlib.util.module_from_spec(spec); spec.loader.exec_module(va)
vs = va.vs
D1_MIN_PAIRS = 12

_crossed = vs.crossed
def crossed_guard(Z, level=0.95, na=vs.N_LAB):
    Z = np.asarray(Z, float)
    if Z.ndim != 2 or Z.shape[1] < 2: return dict(not_estimable=True, n_pairs=int(Z.shape[1]) if Z.ndim == 2 else 0)
    return _crossed(Z, level, na)
vs.crossed = crossed_guard

def find_not_estimable(o, path=""):
    hits = []
    if isinstance(o, dict):
        if o.get("not_estimable") is True: hits.append(path)
        for k, v in o.items(): hits += find_not_estimable(v, f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): hits += find_not_estimable(v, f"{path}[{i}]")
    return hits

def js(o):
    if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [js(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

if __name__ == "__main__":
    models = {}
    for m in va.MODELS:
        d = RES / f"v7_{m}"
        if (d / "metadata.json").exists() and json.load(open(d / "metadata.json", encoding="utf-8")).get("run_complete"): models[m] = va.load_model(d)
    assert len(models) == 5, f"expected 5 complete models, found {sorted(models)}"
    out = va.analyse(models)
    hits = find_not_estimable(out)
    assert hits == ["/S8"], f"an analysis other than S8 met an empty table: {hits}"
    # D1: S8 by the pre-specified fallback rule
    names = list(models); lp = models[names[0]]["meta"]["label_pairs"]; rows = va.class_rows("sensitive", "salient")
    U4 = {m: va.units(models[m]["A"], models[m]["A"].stratum == 4) for m in names}
    kept = {m: int(va.pooled_table([U4[m][U4[m].cond == "none"]], rows, lp)[0].shape[1]) for m in names}
    inc = [m for m in names if kept[m] >= D1_MIN_PAIRS]
    Z8 = va.pooled_table([U4[m][U4[m].cond == "none"] for m in inc], rows, lp)[0] if inc else np.zeros((16, 0))
    est = len(inc) >= va.MIN_MODELS and Z8.shape[1] >= 2
    out["S8"] = dict(deviation="D1 (preregistration_v7.md §11)", frozen_result=out["S8"], pairs_kept_per_model=kept, included_models=inc,
                     excluded_models=[m for m in names if m not in inc], n_pairs=int(Z8.shape[1]), result=_crossed(Z8) if est else None,
                     note=None if est else "not estimable under the complete-pair rule (D1)")
    out["run"] = dict(script="analysis/v7_run.py", frozen_analysis="analysis/v7_analyze.py (unchanged since 18ff241)", deviations=["D1"])
    (RES / "v7_analysis.json").write_text(json.dumps(js(out), indent=1), encoding="utf-8")
    c = out["confirmatory"]; print("rejected:", c["rejected"]); print("Gate 3:", out["gate3"])
