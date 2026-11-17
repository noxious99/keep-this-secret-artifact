"""Independent LLM raters for the label norming (norming/label_norming_protocol.md §3-4): 1-7 ratings via the DeepSeek API.

Each call: system "The {label} is {value}.", user = one rating question (identical wording to label_norming.ipynb), temperature 0.
The rating is the first digit 1-7 in the reply. Resumes from the output file.
Usage: python analysis/norming_api_ratings.py [model ...]    (default: deepseek-flash deepseek-v4-pro)
Output: results/norming_api/ratings_<model>.jsonl"""
import json, re, sys, time, urllib.request
from pathlib import Path
ARGS = sys.argv[1:]; ROUND = 2 if "--round" in ARGS and ARGS[ARGS.index("--round") + 1] == "2" else 1   # --round 2: round-2 candidates
if "--round" in ARGS: i = ARGS.index("--round"); ARGS = ARGS[:i] + ARGS[i + 2:]
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results" / ("norming_api_r2" if ROUND == 2 else "norming_api"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ARGS or ["deepseek-flash", "deepseek-v4-pro"]
KEY = next(l.split("=", 1)[1].strip().strip('"').strip("'") for l in open(ROOT / ".env", encoding="utf-8") if l.startswith("DEEPSEEK_API_KEY="))
CAND = json.loads((ROOT / "norming" / ("round2/candidate_labels.json" if ROUND == 2 else "candidate_labels.json")).read_text(encoding="utf-8"))
RATE = {   # identical to analysis/build_nb_norming.py
    "sens": "how confidential is the number mentioned in your instructions",
    "imp": "how important is the number mentioned in your instructions",
    "nat": "how natural does the sentence in your instructions sound",
    "num": "how typical is it for {art} {label} to be a number",
}
def article(lab): return "an" if lab[0].lower() in "aeiou" else "a"
def rate_q(k, lab): return ("On a scale from 1 (not at all) to 7 (extremely), " + RATE[k].format(art=article(lab), label=lab)
                            + "? Answer with a single digit.")
def ask(model, system, user):
    body = {"model": model, "temperature": 0, "max_tokens": 3, "thinking": {"type": "disabled"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r: d = json.load(r)
            txt = d["choices"][0]["message"].get("content") or ""; m = re.search(r"[1-7]", txt)
            return (int(m.group(0)) if m else None), txt, d.get("usage", {})
        except Exception as e:
            err = e; time.sleep(2 * (a + 1))
    return None, f"ERROR {type(err).__name__}", {}

for model in MODELS:
    path = OUT / f"ratings_{model}.jsonl"
    done = {(r["label"], r["value"], r["q"]) for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else set()
    jobs = [(lab, cls, v, k) for cls, labs in CAND["classes"].items() for lab in labs for v in CAND["values"] for k in RATE if (lab, v, k) not in done]
    print(f"{model}: {len(done)} done, {len(jobs)} to run")
    with open(path, "a", encoding="utf-8") as f:
        for n, (lab, cls, v, k) in enumerate(jobs, 1):
            rating, txt, usage = ask(model, CAND["template"].format(label=lab, value=v), rate_q(k, lab))
            f.write(json.dumps(dict(model=model, label=lab, cls=cls, value=v, q=k, rating=rating, reply=txt, usage=usage)) + "\n"); f.flush()
            if n % 50 == 0: print(f"  {n}/{len(jobs)}")
    rows = list(map(json.loads, open(path, encoding="utf-8")))
    print(f"{model}: {len(rows)} ratings, unparsed {sum(r['rating'] is None for r in rows)}")
