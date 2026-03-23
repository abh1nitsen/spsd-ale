"""
SPSD v4.2 — LLM calls v2 (run_llm_v2.py)
==========================================
Clean single-file LLM runner. Replaces run_llm_calls.py,
run_llm_support.py, run_llm_topup.py.

Reads spsd_results_v2.csv (from run_spsd_v2.py).
For each distilled row (passthrough=False): makes TWO Groq calls:
  1. Raw call   — P\n<original prompt>
  2. Dist call  — ALE distilled packet

Skips passthrough rows entirely (no LLM needed for hypothesis test).
Full resume support — safe to stop and restart at any time.

Setup:
  !pip install -q groq
  Colab Secrets → add GROQ_API_KEY
  Get free key: https://console.groq.com

Run:
  %run run_llm_v2.py

Output: updates /content/spsd_results_v2.csv in place
Next:   %run score_and_test.py
"""

import csv, json, time, os, sys, importlib, random
from pathlib import Path

RESULTS_PATH  = "/content/spsd_results_v2.csv"
DONE_FILE     = "/content/llm_v2_done.json"

GROQ_MODEL    = "llama-3.1-8b-instant"
MAX_TOKENS    = 150
TEMPERATURE   = 0.4
CALL_INTERVAL = 6.0    # seconds between calls (3/min, limit is 30/min)
RETRY_WAIT    = 90     # seconds on rate limit
MAX_RETRIES   = 4

# ── Load modules ──────────────────────────────────────────────────────────
sys.path.insert(0, "/content")
import ale_prompt
importlib.reload(ale_prompt)
ALE_SYSTEM = ale_prompt.ALE_SYSTEM_PROMPT
print(f"ale_prompt loaded | system ~{len(ALE_SYSTEM.split())} words")

# ── Groq setup ────────────────────────────────────────────────────────────
try:
    from groq import Groq
except ImportError:
    os.system("pip install -q groq")
    from groq import Groq

try:
    from google.colab import userdata
    GROQ_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Get free key at https://console.groq.com\n"
        "Add to Colab Secrets (key icon, left sidebar)."
    )

client = Groq(api_key=GROQ_KEY)
print(f"Groq ready | {GROQ_MODEL} | max_tokens={MAX_TOKENS}")

# ── Load results ──────────────────────────────────────────────────────────
if not Path(RESULTS_PATH).exists():
    raise FileNotFoundError(f"Not found: {RESULTS_PATH}\nRun run_spsd_v2.py first.")

with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Only process distilled rows — passthrough rows need no LLM calls
distilled = [r for r in rows if r.get("passthrough") == "False"]
print(f"Loaded {len(rows)} rows | {len(distilled)} distilled | "
      f"{len(rows)-len(distilled)} passthrough (skipped)")

# ── Resume ────────────────────────────────────────────────────────────────
if Path(DONE_FILE).exists():
    with open(DONE_FILE) as f:
        done = set(json.load(f))
    print(f"Resuming — {len(done)} already done")
else:
    done = set()

pending = [r for r in distilled
           if r["id"] not in done
           and not r.get("raw_response","").strip()]

print(f"Pending: {len(pending)} rows\n")

if not pending:
    print("Nothing to do — all distilled rows already have responses.")
    print("Run score_and_test.py directly.")
    exit(0)

def save_done():
    with open(DONE_FILE, "w") as f:
        json.dump(list(done), f)

def est_tokens(text):
    return max(1, round(len(str(text).split()) / 0.75))

ALE_CACHE_READ = round(est_tokens(ALE_SYSTEM) * 0.10)

# ── Groq call ─────────────────────────────────────────────────────────────
def call_groq(system_prompt, user_turn, label=""):
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.5, 2.0))
            t0  = time.time()
            rsp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_turn},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            ms  = (time.time() - t0) * 1000
            txt = rsp.choices[0].message.content or ""
            u   = rsp.usage
            return {"response":      txt,
                    "input_tokens":  u.prompt_tokens,
                    "output_tokens": u.completion_tokens,
                    "latency_ms":    round(ms, 1),
                    "error":         ""}
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n  [429 {label}] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [err {label} a{attempt+1}]: {err[:80]}", flush=True)
                if attempt == MAX_RETRIES - 1:
                    return {"response":"","input_tokens":0,
                            "output_tokens":0,"latency_ms":0,"error":err[:150]}
                time.sleep(10)
    return {"response":"","input_tokens":0,"output_tokens":0,
            "latency_ms":0,"error":"max_retries"}

# ── Main loop ─────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"LLM CALLS — {len(pending)} rows | {GROQ_MODEL}")
print(f"Rate: {CALL_INTERVAL}s/call | est ~{len(pending)*2*CALL_INTERVAL/60:.0f} min")
print(f"{'='*60}\n")

last_call = 0.0
n_done    = 0
n_errors  = 0

for row in pending:
    pid     = row["id"]
    orig    = row.get("raw_prompt", "")
    dist_ut = row.get("ale_user_turn", "")

    print(f"[{len(done)+1:3d}/{len(distilled)}] {pid} "
          f"[{row['category'][:16]:16s}] ", end="", flush=True)

    # Rate limit
    gap = time.time() - last_call
    if gap < CALL_INTERVAL:
        time.sleep(CALL_INTERVAL - gap)

    # Raw call — original prompt in passthrough mode
    raw = call_groq(ALE_SYSTEM, f"P\n{orig}", label=f"{pid}-raw")
    last_call = time.time()

    # Rate limit between paired calls
    time.sleep(CALL_INTERVAL)

    # Distilled call — ALE packet
    dist = call_groq(ALE_SYSTEM, dist_ut, label=f"{pid}-dist")
    last_call = time.time()

    # Token saving
    raw_in   = est_tokens(orig) + 2
    dist_in  = est_tokens(dist_ut)
    save_in  = raw_in - dist_in
    save_out = int(raw["output_tokens"] or 0) - int(dist["output_tokens"] or 0)

    row.update({
        "raw_response":        raw["response"][:500],
        "raw_input_tok_llm":   raw["input_tokens"],
        "raw_output_tokens":   raw["output_tokens"],
        "raw_latency_ms":      raw["latency_ms"],
        "raw_error":           raw["error"],
        "dist_response":       dist["response"][:500],
        "dist_input_tok_llm":  dist["input_tokens"],
        "dist_output_tokens":  dist["output_tokens"],
        "dist_latency_ms":     dist["latency_ms"],
        "dist_error":          dist["error"],
        "token_saving_output": save_out,
        "token_saving_paired": save_in + save_out,
        "token_saving_vs_naive": est_tokens(orig)-ALE_CACHE_READ-dist_in,
        "output_reduction_pct": round(save_out/max(int(raw["output_tokens"] or 1),1)*100,1),
        "llm_model":           GROQ_MODEL,
    })

    status = "OK " if not (raw["error"] or dist["error"]) else "ERR"
    if status == "ERR": n_errors += 1
    print(f"raw={raw['output_tokens']:>3} dist={dist['output_tokens']:>3} "
          f"save_in={save_in:+3d}t [{status}]")

    done.add(pid)
    save_done()
    n_done += 1

    # Save every 10
    if n_done % 10 == 0:
        all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
        with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys,
                               quoting=csv.QUOTE_ALL, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"  [checkpoint {n_done} → {RESULTS_PATH}]")

# ── Final write ───────────────────────────────────────────────────────────
all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_keys,
                       quoting=csv.QUOTE_ALL, restval="")
    w.writeheader()
    w.writerows(rows)

print(f"\n{'='*60}")
print(f"COMPLETE — {n_done} rows | {n_errors} errors")
print(f"Output → {RESULTS_PATH}")
if len(done) >= len(distilled):
    Path(DONE_FILE).unlink(missing_ok=True)
    print("All done — tracker removed")
else:
    print(f"Remaining: {len(distilled)-len(done)} — re-run to continue")
print(f"Next → %run score_and_test.py")
