"""
SPSD v4.2 — Run SPSD on corpus v2 (run_spsd_v2.py)
====================================================
Reads spsd_corpus_v2.csv, runs SPSD on every prompt.
Medical prompts now exit at Tier 1 (domain_medical).
Writes spsd_results_v2.csv.

Run:
  %run run_spsd_v2.py

Output: /content/spsd_results_v2.csv
Next:   %run run_llm_v2.py
"""

import csv, sys, importlib, time
from pathlib import Path

CORPUS_PATH = "/content/spsd_corpus_v2.csv"
OUTPUT_PATH = "/content/spsd_results_v2.csv"
MODEL_PATH  = "/content/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

sys.path.insert(0, "/content")
import spsd_v4, ale_prompt
importlib.reload(spsd_v4)
importlib.reload(ale_prompt)
print(f"spsd_v4 loaded   | SHORT_LIMIT={spsd_v4.SHORT_PROMPT_WORD_LIMIT}")
print(f"ale_prompt loaded")

# Verify v4.2 medical passthrough is active
test_pt, test_r = spsd_v4.tier1_check(
    "A 23-year-old pregnant woman at 22 weeks gestation presents with burning urination")
assert test_pt and test_r == "domain_medical", \
    f"MEDICAL passthrough not working: pt={test_pt} reason={test_r}"
print(f"Medical passthrough check: OK (reason={test_r})")

if not Path(MODEL_PATH).exists():
    print("Downloading model (~986MB)...")
    MODEL_PATH = spsd_v4.download_model(dest_dir="/content/models")

print(f"Loading SLM...")
spsd_v4.load_model(model_path=MODEL_PATH)
print("SLM ready.\n")

if not Path(CORPUS_PATH).exists():
    raise FileNotFoundError(f"Not found: {CORPUS_PATH}\nRun fetch_corpus_v2.py first.")

with open(CORPUS_PATH, newline="", encoding="utf-8") as f:
    corpus = list(csv.DictReader(f))
print(f"Corpus: {len(corpus)} prompts\n")

def est_tokens(text):
    return max(1, round(len(str(text).split()) / 0.75))

ALE_SYSTEM_TOKENS     = est_tokens(ale_prompt.ALE_SYSTEM_PROMPT)
ALE_CACHE_READ_TOKENS = round(ALE_SYSTEM_TOKENS * 0.10)

print(f"{'='*60}")
print(f"SPSD RUN — {len(corpus)} prompts")
print(f"{'='*60}\n")

results     = []
n_dist      = 0
n_pt        = 0
total_saving= 0
t_start     = time.time()

for i, crow in enumerate(corpus, 1):
    prompt = crow["prompt"]
    cat    = crow["category"]

    print(f"[{i:3d}/{len(corpus)}] {crow['id']} [{cat[:18]:18s}] "
          f"({crow['word_count']:>3s}w) ", end="", flush=True)

    distill    = spsd_v4.distill(prompt, model_path=MODEL_PATH)
    ale_packet = ale_prompt.build_ale_messages(distill)
    user_turn  = ale_packet["messages"][0]["content"]
    cx         = distill.complexity

    raw_in  = est_tokens(prompt) + 2
    dist_in = est_tokens(user_turn)

    status = "PASSTHRU" if distill.passthrough else \
             f"DISTILL saving={distill.token_saving:+d}t"
    print(f"→ {status} ({distill.latency_ms:.0f}ms)")

    if distill.passthrough: n_pt += 1
    else:
        n_dist += 1
        total_saving += distill.token_saving

    results.append({
        "id":                  crow["id"],
        "category":            crow["category"],
        "word_count":          crow["word_count"],
        "source":              crow.get("source",""),
        "intent_label":        crow.get("intent_label",""),
        "domain_tag":          crow.get("domain_tag",""),
        "passthrough":         distill.passthrough,
        "passthrough_reason":  distill.passthrough_reason or "",
        "distill_tier":        distill.tier,
        "distill_latency_ms":  round(distill.latency_ms, 1),
        "domain":              distill.domain or "",
        "complexity_profile":  cx.profile if cx else "",
        "social_score":        round(cx.social_score, 3) if cx else "",
        "semantic_score":      round(cx.semantic_score, 3) if cx else "",
        "structural_score":    round(cx.structural_score, 3) if cx else "",
        "repetition_score":    round(cx.repetition_score, 3) if cx else "",
        "rec_ratio":           round(cx.recommended_ratio, 3) if cx else "",
        "confidence":          round(distill.confidence, 3),
        "token_saving_spsd":   distill.token_saving,
        "hfg_aux":             " | ".join(distill.hfg_aux or []),
        "compressed_prompt":   distill.compressed_prompt[:400],
        "ale_user_turn":       user_turn,
        "raw_prompt":          prompt[:400],
        "raw_input_tokens":    raw_in,
        "dist_input_tokens":   dist_in,
        "system_tokens":       ALE_SYSTEM_TOKENS,
        "ale_cache_read_tokens": ALE_CACHE_READ_TOKENS,
        "token_saving_input":  raw_in - dist_in,
        "token_saving_pct":    round((raw_in-dist_in)/max(raw_in,1)*100, 1),
        "token_saving_vs_naive": est_tokens(prompt)-ALE_CACHE_READ_TOKENS-dist_in,
        "break_even_met":      (est_tokens(prompt)-ALE_CACHE_READ_TOKENS-dist_in)>0,
        "compression_ratio":   round(dist_in/max(raw_in,1), 3),
        # LLM columns — filled by run_llm_v2.py
        "raw_response":        "",
        "raw_input_tok_llm":   "",
        "raw_output_tokens":   "",
        "raw_latency_ms":      "",
        "raw_error":           "",
        "dist_response":       "",
        "dist_input_tok_llm":  "",
        "dist_output_tokens":  "",
        "dist_latency_ms":     "",
        "dist_error":          "",
        "token_saving_output": "",
        "token_saving_paired": "",
        "output_reduction_pct":"",
        "llm_model":           "",
        "semantic_similarity": "",
        "quality_flag":        "",
    })

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()),
                       quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(results)

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"SPSD COMPLETE — {len(results)} prompts in {elapsed/60:.1f} min")
print(f"Distilled   : {n_dist} | Passthrough: {n_pt}")
print(f"Total saving: {total_saving:+d} tokens")
if n_dist: print(f"Avg/distill : {total_saving/n_dist:.1f} tokens")
print(f"Output → {OUTPUT_PATH}")
print(f"Next   → %run run_llm_v2.py")
