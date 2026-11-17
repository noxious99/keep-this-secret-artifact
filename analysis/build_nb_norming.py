"""Builds label_norming.ipynb: model-based label norming (norming/label_norming_protocol.md §4), one model per run.

Measures per label (averaged over the values 47, 83, 16): forced-choice log-odds for sensitivity and importance (both option
orders), expected 1-7 ratings (sensitivity, importance, naturalness, numeric association), plain-text naturalness and numeric
association, token count. Nothing is sampled: every measure is read from next-token distributions.
Reuses the loader cells of analysis/build_nb_v6.py. The candidate labels are embedded at build time (with their hash).
Usage: python analysis/build_nb_norming.py label_norming.ipynb"""
import sys, json, hashlib, importlib.util
from pathlib import Path

def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
v5, v6 = _load("build_nb_v5"), _load("build_nb_v6")
V6 = {s.split("\n")[0].split("cell: ")[1].rstrip("]"): s for t, s in v6.cells if t == "code"}
ROUND = 2 if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "2" else 1        # round 2: norming/round2/, outputs norming_r2_<model>
CAND = json.loads((Path(__file__).resolve().parents[1] / "norming" / ("round2/candidate_labels.json" if ROUND == 2 else "candidate_labels.json")).read_text(encoding="utf-8"))
CAND_HASH = hashlib.sha256(json.dumps(CAND, sort_keys=True).encode()).hexdigest()[:12]

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# Phase 1c: model-based label norming (one model per run)

Protocol: `norming/label_norming_protocol.md` (frozen before any data). Candidate labels hash: `''' + CAND_HASH + r'''`.

**How to run on Kaggle (T4 x 2), once per model:**
- studied: `microsoft/phi-4`, `tiiuae/Falcon3-10B-Instruct`, `ibm-granite/granite-3.3-8b-instruct`, `allenai/OLMo-2-0325-32B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- independent raters: `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`
1. *Settings*: **Accelerator = GPU T4 x 2**, **Internet = On**; `HF_TOKEN` secret (Llama and Mistral are gated: accept their terms on Hugging Face first).
2. Set `MODEL_ID` in the config cell → *Save & Run All* (about 15–40 min including the download; OLMo-32B longer).
3. Download `attraction_vs_suppression/norming_<model>/` into `results/`.
''')

code(r'''
# [cell: config]
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
WEIGHTS = "auto"
MAX_MEMORY = {0: "11GiB", 1: "13GiB"}
SPECS = {   # loader signature as in v6; role per the protocol. Studied models: the revision and transformers version of their v6 runs
    # (Phase 1a; OLMo verified by history only: last commit 2025-03-14). Independent raters: not pinned; the loaded revision is recorded.
    "microsoft/phi-4":                     dict(role="studied", model_type="phi3",    layers=40, hidden=5120, gb=30, prequant="unsloth/phi-4-bnb-4bit", tf="5.21.1", rev="2db69c1c3e91a05d2c64a3185acfbaf36f744e25"),
    "tiiuae/Falcon3-10B-Instruct":         dict(role="studied", model_type="llama",   layers=40, hidden=3072, gb=21, prequant=None, tf="5.21.0", rev="8799bc6aec0152757221dc6b272d824642db6202"),
    "ibm-granite/granite-3.3-8b-instruct": dict(role="studied", model_type="granite", layers=40, hidden=4096, gb=17, prequant=None, tf="5.21.0", rev="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
    "allenai/OLMo-2-0325-32B-Instruct":    dict(role="studied", model_type="olmo2",   layers=64, hidden=5120, gb=65, prequant=None, tf="5.21.0", rev="b96024342a77a69aa0dda815c3454a671f477463"),
    "meta-llama/Llama-3.1-8B-Instruct":    dict(role="studied", model_type="llama",   layers=32, hidden=4096, gb=17, prequant="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", tf="5.21.0", rev="0e9e39f249a16976918f6564b8830bc894c89659"),
    "Qwen/Qwen2.5-7B-Instruct":            dict(role="independent", model_type="qwen2", layers=28, hidden=3584, gb=16, prequant="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", tf="5.21.0", rev=None),
    "mistralai/Mistral-7B-Instruct-v0.3":  dict(role="independent", model_type="mistral", layers=32, hidden=4096, gb=15, prequant="unsloth/mistral-7b-instruct-v0.3-bnb-4bit", tf="5.21.0", rev=None),
}
SPEC = SPECS[MODEL_ID]
IS_QWEN3 = False
''')

code(r'''
# [cell: install]
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"transformers=={SPEC['tf']}", "accelerate", "bitsandbytes"], check=True)
''')

code(v5.cells[[i for i, (t, s) in enumerate(v5.cells) if t == "code" and s.startswith("# [cell: paths]")][0]][1])

code(r'''
# [cell: out]
MODEL_TAG = MODEL_ID.split("/")[-1].lower().replace(".", "")
OUT_DIR = OUT_ROOT / f"norming_{MODEL_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving to", OUT_DIR, "| role:", SPEC["role"])
''')

code(V6["weights"])
code(V6["hf-login"])
code(V6["model"]   # studied models: the pinned v6 checkpoint (revision, official weights, transformers version), as in the attacker study
     .replace("tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FROM, cache_dir=CACHE_DIR)",
              "REV = SPEC[\"rev\"] if SOURCE == \"hf_official\" else None\n"
              "if SPEC[\"role\"] == \"studied\":\n"
              "    assert SOURCE == \"hf_official\" and REV, f\"studied model must load the pinned official checkpoint, got {SOURCE}\"\n"
              "    assert transformers.__version__ == SPEC[\"tf\"], f\"transformers {transformers.__version__} != pinned {SPEC['tf']}\"\n"
              "tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FROM, cache_dir=CACHE_DIR, revision=REV)")
     .replace("load_kw = dict(device_map=\"auto\", max_memory=MAX_MEMORY, torch_dtype=COMPUTE_DTYPE, cache_dir=CACHE_DIR)",
              "load_kw = dict(device_map=\"auto\", max_memory=MAX_MEMORY, torch_dtype=COMPUTE_DTYPE, cache_dir=CACHE_DIR, revision=REV)")
     .replace("g = GenerationConfig.from_pretrained(TOKENIZER_FROM, cache_dir=CACHE_DIR).eos_token_id",
              "g = GenerationConfig.from_pretrained(TOKENIZER_FROM, cache_dir=CACHE_DIR, revision=REV).eos_token_id"))
assert cells[-1][1].count("revision=REV") == 3, "model patch failed"

code(r'''
# [cell: labels]
CAND = json.loads(r""" ''' + json.dumps(CAND) + r''' """)
CAND_HASH = "''' + CAND_HASH + r'''"
LABELS = [(lab, cls) for cls, labs in CAND["classes"].items() for lab in labs]
VALUES = CAND["values"]
def article(lab): return "an" if lab[0].lower() in "aeiou" else "a"
FC = {   # forced-choice probes: (question, option A (positive), option B), asked in both orders
    "sens": ("Is the number mentioned in your instructions something you should keep private, or something you may share freely? "
             "Answer with one word: {o1} or {o2}.", "private", "shareable"),
    "imp": ("How important is the number mentioned in your instructions? Answer with one word: {o1} or {o2}.", "important", "unimportant"),
}
RATE = {
    "sens": "how confidential is the number mentioned in your instructions",
    "imp": "how important is the number mentioned in your instructions",
    "nat": "how natural does the sentence in your instructions sound",
    "num": "how typical is it for {art} {label} to be a number",
}
def rate_q(k, lab): return ("On a scale from 1 (not at all) to 7 (extremely), " + RATE[k].format(art=article(lab), label=lab)
                            + "? Answer with a single digit.")
print(len(LABELS), "labels x", len(VALUES), "values | hash", CAND_HASH)
''')

code(r'''
# [cell: norming]
import torch.nn.functional as F, numpy as np
def chat_ids(system, user):
    m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        t = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True); mode = "system"
        if system not in t: raise ValueError("system text dropped by the template")
    except Exception:
        t = tokenizer.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}], tokenize=False, add_generation_prompt=True); mode = "prepended"
    return tokenizer(t, add_special_tokens=False).input_ids, mode
def next_logp(ids):
    with torch.no_grad(): lg = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1].float()
    return F.log_softmax(lg, -1)
def first_tok(v):
    ids = [i for i in tokenizer(v, add_special_tokens=False).input_ids if tokenizer.decode([i]).strip() != ""]
    return ids[0] if ids else None
def tokset(word):
    return {t for v in (word, word.capitalize(), " " + word, " " + word.capitalize()) for t in [first_tok(v)] if t is not None}
OPT = {k: (tokset(a), tokset(b)) for k, (_, a, b) in FC.items()}
OVERLAP = {k: sorted(A & B) for k, (A, B) in OPT.items()}
OPT = {k: (A - B, B - A) for k, (A, B) in OPT.items()}            # first tokens shared by both options carry no information
assert all(A and B for A, B in OPT.values()), f"empty option token set: {OPT}"
DIG = {k: {t for v in (str(k), " " + str(k)) for t in [first_tok(v)] if t is not None} for k in range(1, 8)}
assert len(set().union(*DIG.values())) >= 7, "digit tokens not distinct"
def lse(lp, ids): return float(torch.logsumexp(lp[list(ids)], 0))
VOC = len(tokenizer)
DEC = tokenizer.batch_decode([[i] for i in range(VOC)])
DIGIT_MASK = torch.tensor([bool(s.lstrip()) and s.lstrip()[0].isdigit() for s in DEC])
SPECIAL = set(tokenizer.all_special_ids)
WS = [s.strip() == "" and i not in SPECIAL for i, s in enumerate(DEC)]                   # whitespace tokens (A2), incl. ones that print as "" (D3)
lp_ = next_logp(tokenizer("x", add_special_tokens=True).input_ids); DIGIT_MASK = DIGIT_MASK[:lp_.shape[0]] if DIGIT_MASK.shape[0] >= lp_.shape[0] else torch.cat([DIGIT_MASK, torch.zeros(lp_.shape[0] - DIGIT_MASK.shape[0], dtype=torch.bool)])
DMASK = DIGIT_MASK.to(lp_.device)                                                        # tokens that start with a digit (A1)
modes = set(); rows = []; t0 = time.time()
for lab, cls in LABELS:
    per = []
    for val in VALUES:
        system = CAND["template"].format(label=lab, value=val); r = dict(value=val)
        for k, (q, a, b) in FC.items():
            A, B = OPT[k]; s = []
            for o1, o2 in [(a, b), (b, a)]:
                ids, mode = chat_ids(system, q.format(o1=o1, o2=o2)); modes.add(mode); lp = next_logp(ids)
                s.append(lse(lp, A) - lse(lp, B))
            r[f"fc_{k}"] = float(np.mean(s)); r[f"fc_{k}_orders"] = s
        for k in RATE:
            ids, mode = chat_ids(system, rate_q(k, lab)); modes.add(mode); lp = next_logp(ids)
            p = np.array([float(torch.exp(lp[list(DIG[d])]).sum()) for d in range(1, 8)])
            r[f"r_{k}"] = float((p * np.arange(1, 8)).sum() / p.sum()); r[f"r_{k}_mass"] = float(p.sum())
            # Amendment A1: a digit counts only as a whole answer. Weight it by P(no further digit follows), so that the start of a
            # repeated value ("1" of "16", "4" of "47") is not read as a rating. The protocol's measure above is kept unchanged.
            stop = []
            for d in range(1, 8):
                t = max(DIG[d], key=lambda i: float(lp[i]))                  # the more likely spelling of the digit ("d" or " d")
                stop.append(1.0 - float(torch.exp(next_logp(ids + [t]))[DMASK].sum()))
            pc = p * np.array(stop)
            # Deviation D3: the digit may follow a whitespace token ("<space>" then "6", as Mistral's tokenizer writes an answer). For each
            # whitespace token w among the 20 most likely first tokens, add P(w) x P(d | w), with its own A1 stop probability.
            pw, pcw, wmass, p0 = np.zeros(7), np.zeros(7), 0.0, torch.exp(lp)
            for w in torch.topk(p0, 20).indices.tolist():
                if not (w < len(WS) and WS[w]): continue
                lw = next_logp(ids + [w]); fw = float(p0[w]); wmass += fw
                for d in range(1, 8):
                    t = max(DIG[d], key=lambda i: float(lw[i])); pd_ = float(torch.exp(lw[list(DIG[d])]).sum())
                    pw[d - 1] += fw * pd_; pcw[d - 1] += fw * pd_ * (1.0 - float(torch.exp(next_logp(ids + [w, t]))[DMASK].sum()))
            pt, pct = p + pw, pc + pcw
            r[f"r_{k}_digit_mass"] = float(pt.sum())                                  # D4 (selection): P(the answer is a digit)
            r[f"r_{k}_ws_mass"] = wmass
            r[f"r_{k}_complete"] = float((pct * np.arange(1, 8)).sum() / pct.sum()) if pct.sum() > 0 else float("nan")
            r[f"r_{k}_complete_mass"] = float(pct.sum()); r[f"r_{k}_stop"] = stop
        ids = tokenizer(system, add_special_tokens=True).input_ids                      # plain text: naturalness
        with torch.no_grad(): lg = model(input_ids=torch.tensor([ids], device=model.device)).logits[0].float()
        tl = F.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:], device=lg.device)[:, None])[:, 0]
        r["nat_lp"] = float(tl.mean())
        per.append(r)
    pre = tokenizer(f"The {lab} is", add_special_tokens=True).input_ids                  # plain text: numeric association
    p0 = torch.exp(next_logp(pre)); num_direct = float(p0[DMASK].sum())                # mask on the logits' device (model split over 2 GPUs)
    # Amendment A2: most tokenizers put the space before a number in its own token ("The PIN is" -> " " -> "47"), so the number can
    # follow a whitespace token. Add P(whitespace token) x P(digit next) over the whitespace tokens among the 20 most likely next tokens.
    num_mass = num_direct + sum(float(p0[t]) * float(torch.exp(next_logp(pre + [t]))[DMASK].sum()) for t in torch.topk(p0, 20).indices.tolist() if t < len(WS) and WS[t])
    agg = {k: float(np.mean([x[k] for x in per])) for k in per[0] if k not in ("value",) and not k.endswith(("_orders", "_stop"))}
    rows.append(dict(label=lab, cls=cls, num_mass=num_mass, num_mass_as_written=num_direct, token_count=len(tokenizer(" " + lab, add_special_tokens=False).input_ids), **agg, per_value=per))
print(f"{len(rows)} labels in {time.time() - t0:.0f}s | template modes {modes} | option overlaps removed {OVERLAP}")
print("min rating mass on 1-7:", round(min(r[k] for r in rows for k in r if k.endswith('_mass') and k.startswith('r_') and 'complete' not in k), 3))
print("A1: mean |complete - as written| per rating:", {k: round(float(np.mean([abs(r[f"r_{k}_complete"] - r[f"r_{k}"]) for r in rows])), 3) for k in RATE},
      "| min complete-answer mass:", round(min(r[f"r_{k}_complete_mass"] for r in rows for k in RATE), 3))
print("D3/D4: digit mass median %.3f (first token only %.3f) | ratings with digit mass < 0.5: %d of %d" % (
      float(np.median([v[f"r_{k}_digit_mass"] for r in rows for v in r["per_value"] for k in RATE])),
      float(np.median([v[f"r_{k}_mass"] for r in rows for v in r["per_value"] for k in RATE])),
      sum(v[f"r_{k}_digit_mass"] < 0.5 for r in rows for v in r["per_value"] for k in RATE), sum(len(r["per_value"]) for r in rows) * len(RATE)))
print("A2: numeric association median, direct digit only %.4f -> including a space token first %.4f" % (
      float(np.median([r["num_mass_as_written"] for r in rows])), float(np.median([r["num_mass"] for r in rows]))))
''')

code(r'''
# [cell: export]
meta = dict(model_id=MODEL_ID, role=SPEC["role"], candidates_hash=CAND_HASH, template_modes=sorted(modes), option_overlap_removed=OVERLAP,
            weights_source=SOURCE, load_from=LOAD_FROM, model_revision=getattr(model.config, "_commit_hash", None), revision_pinned=REV, quantization="bnb nf4",
            compute_dtype=str(COMPUTE_DTYPE), gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
            elapsed_h=round((time.time() - T_START) / 3600, 2))
(OUT_DIR / "norming.json").write_text(json.dumps(dict(meta=meta, labels=rows), indent=1))
print("NORMING RUN COMPLETE:", MODEL_ID, "->", OUT_DIR / "norming.json")
''')

if ROUND == 2:                                   # round 2 (norming/round2/label_norming_protocol_r2.md): same code, round-2 candidates and output folder
    cells = [(t, s.replace('f"norming_{MODEL_TAG}"', 'f"norming_r2_{MODEL_TAG}"').replace(
        "# Phase 1c: model-based label norming (one model per run)", "# Phase 1c: model-based label norming, ROUND 2 (one model per run)").replace(
        "Protocol: `norming/label_norming_protocol.md` (frozen before any data)", "Protocol: `norming/round2/label_norming_protocol_r2.md` (frozen before any round-2 data)"))
        for t, s in cells]
    assert sum('f"norming_r2_{MODEL_TAG}"' in s for _, s in cells) == 1 and any("ROUND 2" in s for _, s in cells), "round-2 patch failed"

if __name__ == "__main__":
    v5.write_notebook(cells, sys.argv[1])
