SPSD v4.2 — Replication Guide
Sentiment Preserving Semantic Distillation  + Adaptive Logic Engine
This guide covers everything needed to replicate the full pipeline from scratch:
corpus fetch → SPSD distillation → LLM evaluation → hypothesis testing → paper.
---
Contents
Overview
File Inventory
Prerequisites
Environment Setup
Step-by-Step Replication
Configuration Reference
Frontier LLM Options
Expected Outputs
Troubleshooting
Architecture Summary
---
1. Overview
SPSD is an on-device prompt compression pipeline. Before a user prompt reaches
a frontier LLM (GPT-4o, Llama-3, Claude, etc.), a 4-bit quantized 1.5B SLM
running on the edge device compresses it, stripping social scaffolding and
preserving semantic content. The frontier LLM receives a compact structured
packet and reconstructs a full natural-language response.
Core claim (empirically validated):
Mean input token saving: 62.3 tokens per distilled call (t=16.17, p<0.001, d=1.92)
Response quality: mean cosine similarity 0.763 vs 0.70 threshold (p<0.001)
All 71 distilled calls produced positive savings (100%)
The pipeline has no dependency on any specific frontier LLM provider.
The evaluation used Groq (free tier). The `round_trip()` demo function supports
Groq, Anthropic, and OpenAI.
---
2. File Inventory
Core pipeline (required)
File	Purpose
`spsd_v4.py`	Main SPSD pipeline — Tier 1 gate, complexity scorer, HFG, SLM, economic gates
`ale_prompt.py`	ALE packet builder — formats distilled output for frontier LLM
Corpus and evaluation scripts (run in order)
File	Step	Input	Output
`fetch_corpus_v2.py`	1	HuggingFace API	`spsd_corpus_v2.csv`
`run_spsd_v2.py`	2	`spsd_corpus_v2.csv`	`spsd_results_v2.csv`
`run_llm_v2.py`	3	`spsd_results_v2.csv`	updates `spsd_results_v2.csv`
`score_and_test.py`	4	`spsd_results_v2.csv`	`spsd_hypothesis_v2.xlsx`
Legacy scripts (from earlier runs — not needed for fresh replication)
File	Notes
`fetch_support_only.py`	Generates 32 synthetic support prompts (now included in corpus v2)
`run_spsd_only.py`	Older SPSD runner — use `run_spsd_v2.py` instead
`run_llm_support.py`	Older LLM runner — use `run_llm_v2.py` instead
`run_llm_topup.py`	Targeted re-run for missing rows — use `run_llm_v2.py` instead
`merge_and_score.py`	Older merger — use `score_and_test.py` instead
`score_quality.py`	Standalone similarity scorer — now integrated into `score_and_test.py`
`run_llm_calls.py`	Deprecated — used Gemini (saturated). Do not use.
Results
File	Contents
`spsd_hypothesis_final.xlsx`	Full hypothesis test results (5 sheets)
	
---
3. Prerequisites
Accounts and API keys
Service	Required for	Cost	Get key at
Groq	LLM evaluation calls	Free (rate-limited)	console.groq.com
Google Colab	Running the pipeline	Free tier sufficient	colab.research.google.com
HuggingFace	Corpus fetch (no auth needed)	Free, no account	—
Anthropic	Optional demo only	Paid	console.anthropic.com
OpenAI	Optional demo only	Paid	platform.openai.com
The full replication pipeline uses only Groq (free tier) and Google Colab
(free tier). No paid API access is required.
Groq free tier limits
Limit	Value
Requests/day	14,400
Tokens/minute (llama-3.1-8b-instant)	~20,000
Rate limit behaviour	HTTP 429, retry after 90s
At 6 seconds between calls and 71 distilled prompts requiring 2 calls each
(raw + distilled), the full LLM evaluation takes approximately 15–20 minutes.
---
4. Environment Setup
4.1 Google Colab setup (recommended for full replication)
```python
# Cell 1 — Install llama-cpp-python (pre-built CPU wheel, ~2 min)
!pip install -q llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Cell 2 — Install other dependencies
!pip install -q groq sentence-transformers huggingface_hub \
    scipy openpyxl matplotlib requests

# Cell 3 — Add Groq API key to Colab Secrets
# In Colab: click the key icon (🔑) in the left sidebar
# Add secret name: GROQ_API_KEY
# Value: your Groq API key from console.groq.com
```
4.2 Local setup (alternative)
```bash
# Python 3.10+ required
pip install llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install groq sentence-transformers huggingface_hub \
    scipy openpyxl matplotlib requests

# Set environment variable
export GROQ_API_KEY="your_groq_key_here"
```
4.3 Download the SLM model
```python
# Run once — downloads ~986 MB to /content/models/
# (In Colab, save to Google Drive to persist across sessions)
import sys
sys.path.insert(0, '/content')
import spsd_v4
MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')
print(f"Model at: {MODEL_PATH}")

# Optional: save to Drive so you don't re-download each session
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd/models', exist_ok=True)
shutil.copy(MODEL_PATH, '/content/drive/MyDrive/spsd/models/')
```
---
5. Step-by-Step Replication
Upload `spsd_v4.py` and `ale_prompt.py` to `/content/` before running any step.
Critical: always reload modules after upload
```python
import sys, importlib

# Purge any cached versions
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules[key]

import spsd_v4, ale_prompt
importlib.reload(spsd_v4)
importlib.reload(ale_prompt)
print(f"spsd_v4 loaded   | SHORT_LIMIT={spsd_v4.SHORT_PROMPT_WORD_LIMIT}")
print(f"ale_prompt loaded | {len(ale_prompt.ALE_SYSTEM_PROMPT.split())} words in system prompt")

# Verify v4.2 medical passthrough is active
test_prompt = (
    "A 23-year-old pregnant woman at 22 weeks gestation presents to her physician "
    "with burning urination that started yesterday and she is concerned about "
    "treatment safety during pregnancy"
)
pt, reason = spsd_v4.tier1_check(test_prompt)
assert pt and reason == "domain_medical", \
    f"Medical passthrough not working: pt={pt} reason={reason}"
print(f"Medical passthrough: OK (reason={reason})")

# Verify code domain is tag-only (not passthroughed)
code_test = (
    "I keep getting a NullPointerException in my Java checkout service every time "
    "a user applies a discount code the stack trace shows CartService line 42"
)
pt2, reason2 = spsd_v4.tier1_check(code_test)
assert not pt2, f"Code should NOT passthrough but got pt={pt2} reason={reason2}"
print(f"Code tag-only: OK (pt={pt2})")
```
---
Step 1 — Fetch corpus (~5 min)
Upload `fetch_corpus_v2.py` to `/content/`, then:
```python
%run fetch_corpus_v2.py
```
Output: `/content/spsd_corpus_v2.csv` — 150 prompts across 7 categories.
What it fetches:
Category	n	Source
verbose_social	25	Synthetic (built into script)
general_conversational	30	WildChat-4.8M (HuggingFace, no auth)
code_technical	25	CodeFeedback-Filtered-Instruction (HuggingFace)
multi_intent_linked	25	WildChat-4.8M + UltraChat_200k (HuggingFace)
single_intent_clear	15	Synthetic (built into script)
high_stakes_medical	15	MedQA-USMLE-4-options (HuggingFace)
short_passthrough	15	Bitext customer support (HuggingFace)
If a HuggingFace dataset is unavailable, the script will print a fetch error
and continue. Categories with fewer than their target count will be noted in the
summary. The pipeline tolerates a corpus of 130–150 prompts without issue.
---
Step 2 — Run SPSD distillation (~30–45 min on Colab CPU)
Upload `run_spsd_v2.py` to `/content/`, then:
```python
%run run_spsd_v2.py
```
Output: `/content/spsd_results_v2.csv`
What to expect:
Medical prompts → `PASSTHRU reason=domain_medical`
Short prompts → `PASSTHRU reason=short_prompt`
Distilled prompts → `DISTILL saving=+XXt (YYYms)`
Typical saving range: +22 to +146 tokens
Mean latency per SLM call: 80–180ms (CPU), 50–100ms (NPU)
Save to Drive after this step:
```python
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd', exist_ok=True)
shutil.copy('/content/spsd_results_v2.csv',
            '/content/drive/MyDrive/spsd/spsd_results_v2.csv')
print("Saved")
```
---
Step 3 — Run LLM calls (~15–20 min)
Upload `run_llm_v2.py` to `/content/`. Ensure `GROQ_API_KEY` is set in Colab
Secrets (left sidebar → 🔑 icon → add key). Then:
```python
%run run_llm_v2.py
```
What it does:
Reads all `passthrough=False` rows from `spsd_results_v2.csv`
For each row: makes a raw call (`P\n<original>`) and a distilled call (ALE packet)
Skips passthrough rows entirely — no LLM calls needed for those
Saves a completion tracker `llm_v2_done.json` — safe to stop and restart
Checkpoints every 10 rows
If interrupted: simply re-run `%run run_llm_v2.py` — it resumes from last checkpoint.
To force a full re-run (e.g., after changing the corpus):
```python
import os
for f in ['/content/llm_v2_done.json']:
    if os.path.exists(f): os.remove(f)
# Also clear old responses from spsd_results_v2.csv if needed:
import csv
with open('/content/spsd_results_v2.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row.get('passthrough') == 'False':
        for col in ['raw_response','dist_response','raw_output_tokens',
                    'dist_output_tokens','llm_model','semantic_similarity','quality_flag']:
            row[col] = ''
# Write back (add all_keys logic as in the script)
```
What to expect in output:
```
[ 1/71] P001 [verbose_social   ] raw=127 dist=116 save_in=+77t [OK ]
[ 2/71] P002 [verbose_social   ] raw=134 dist=108 save_in=+84t [OK ]
...
```
`save_in` is the input token saving (positive = SPSD saved tokens).
`raw` and `dist` are output token counts from the LLM (secondary metric, not in H1).
---
Step 4 — Score quality + hypothesis tests
Upload `score_and_test.py` to `/content/`, then:
```python
%run score_and_test.py
```
What it does:
Scores cosine similarity on all paired response rows using `all-MiniLM-L6-v2`
Writes similarity scores back to `spsd_results_v2.csv`
Runs T1 (token saving t-test), T2 (quality t-test), T3 (Kruskal-Wallis ANOVA)
Generates 4 charts
Saves `spsd_hypothesis_v2.xlsx` (5 sheets)
Expected results:
```
T1: t=16.17 p=1.23e-25 d=1.92 → REJECT H0 ✓
T2: t=4.22  p=0.000038 d=0.52 → REJECT H0 ✓
T3: H=4.96  p=0.175          → NOT significant (negative result)
```
---
Step 5 — Save everything to Drive
```python
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd', exist_ok=True)

for fname in ['spsd_corpus_v2.csv', 'spsd_results_v2.csv',
              'spsd_hypothesis_v2.xlsx']:
    shutil.copy(f'/content/{fname}',
                f'/content/drive/MyDrive/spsd/{fname}')
    print(f"Saved: {fname}")
```
---
6. Configuration Reference
spsd_v4.py — key constants
Constant	Value	Meaning
`SHORT_PROMPT_WORD_LIMIT`	15	Prompts ≤15 words always passthrough
`ALE_HEADER_OVERHEAD_TOKENS`	6	Token cost of D|tone|urgency|aux header
`MIN_NET_TOKEN_SAVING`	10	Minimum saving to justify distillation
`SLM_TEMPERATURE`	0.1	SLM inference temperature (low = deterministic)
Dynamic confidence thresholds (tier1_check)
Prompt length	Confidence threshold
≤ 30 words	0.80
≤ 60 words	0.72
> 60 words	0.65
Passthrough triggers (in order)
`short_prompt` — word count ≤ 15
`safety_critical` — hard safety phrases (chest pain, overdose, 911...)
`domain_medical` — 65 clinical keywords + clinical structure regex
`domain_legal` — 30 legal keywords/phrases
`complexity_structural_density_X.XX` — structural score ≥ 0.22 (code/math)
`low_confidence_X.XX_threshold_X.XX` — SLM confidence below threshold
`no_token_saving_±Xt` — net saving below MIN_NET_TOKEN_SAVING
run_llm_v2.py — key constants
Constant	Value	Meaning
`GROQ_MODEL`	`llama-3.1-8b-instant`	Frontier model for evaluation
`MAX_TOKENS`	150	Max output tokens per call
`TEMPERATURE`	0.4	LLM temperature
`CALL_INTERVAL`	6.0s	Seconds between API calls (rate limit buffer)
`RETRY_WAIT`	90s	Wait on HTTP 429 (rate limit)
`MAX_RETRIES`	4	Max retries per call before recording error
---
7. Frontier LLM Options
The evaluation pipeline uses Groq (free tier). The `round_trip()` demo
function in `ale_prompt.py` supports three providers. To use a different
frontier LLM in the evaluation, change `GROQ_MODEL` in `run_llm_v2.py`
to any Groq-hosted model, or adapt the call_groq function for your provider.
Groq models (recommended for replication)
Model	Notes
`llama-3.1-8b-instant`	Used in evaluation. Fast, high TPM limit.
`llama-3.3-70b-versatile`	Higher quality, lower TPM limit (~6,000/min)
`mixtral-8x7b-32768`	Alternative, good quality
Anthropic (optional, paid)
```python
from ale_prompt import round_trip
result = round_trip(
    prompt="Your prompt here",
    api_key="YOUR_ANTHROPIC_KEY",
    model="claude-sonnet-4-6",   # or claude-opus-4-6, claude-haiku-4-5
    provider="anthropic"
)
result.display()
```
OpenAI (optional, paid)
```python
result = round_trip(
    prompt="Your prompt here",
    api_key="YOUR_OPENAI_KEY",
    model="gpt-4o",
    provider="openai"
)
result.display()
```
Using SPSD without any cloud LLM (distillation only)
```python
import sys
sys.path.insert(0, '/path/to/spsd')
import spsd_v4

spsd_v4.load_model(model_path='/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf')
result = spsd_v4.distill("Your prompt here")
print(result.summary())
print(f"Token saving: {result.token_saving}")
print(f"Passthrough: {result.passthrough} ({result.passthrough_reason})")
```
---
8. Expected Outputs
spsd_corpus_v2.csv
150 rows, 7 columns: `id, category, word_count, prompt, source, intent_label, domain_tag`
spsd_results_v2.csv
150 rows, 48 columns including:
SPSD output: `passthrough, passthrough_reason, complexity_profile, token_saving_input, compression_ratio, confidence, hfg_aux, ale_user_turn`
LLM responses: `raw_response, dist_response, raw_output_tokens, dist_output_tokens`
Quality: `semantic_similarity, quality_flag`
spsd_hypothesis_v2.xlsx
5 sheets:
Executive Summary — KPI cards + hypothesis table + key findings
Token Saving — T1 descriptive stats + by-category table + 2 charts
Quality Analysis — T2 stats + by-category table + exclusion note + 2 charts
Full Data — all 150 rows colour-coded by passthrough/quality/category
Methodology — corpus, model, tests, limitations
Routing summary (approximate, varies by corpus fetch)
Outcome	Count	%
Distilled	71	47.3%
Passthrough (all reasons)	79	52.7%
— domain_medical	17	11.3%
— short_prompt	19	12.7%
— no_token_saving	35	23.3%
— domain_legal	3	2.0%
— other	5	3.3%
---
9. Troubleshooting
"MEDICAL passthrough not working" assertion fails on run_spsd_v2.py
The test prompt in the assertion is under 15 words. This is a test prompt issue,
not a pipeline issue. Patch directly in Colab:
```python
content = open("/content/run_spsd_v2.py").read()
content = content.replace(
    '"A 23-year-old pregnant woman at 22 weeks gestation presents with burning urination"',
    '"A 23-year-old pregnant woman at 22 weeks gestation presents to her physician '
    'with burning urination that started yesterday and she is concerned about '
    'treatment safety during pregnancy"'
)
open("/content/run_spsd_v2.py", "w").write(content)
%run run_spsd_v2.py
```
Medical prompts still getting distilled despite fix
The old module is cached in memory. Run the full purge:
```python
import sys, importlib
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules[key]
import spsd_v4, ale_prompt
importlib.reload(spsd_v4)
importlib.reload(ale_prompt)
```
Then verify: `spsd_v4.tier1_check("A patient presents to the physician with chest symptoms")` must return `(True, 'domain_medical')`.
LLM run resumes from wrong position
Delete the completion tracker:
```python
import os
if os.path.exists('/content/llm_v2_done.json'):
    os.remove('/content/llm_v2_done.json')
```
Groq 429 rate limit
`run_llm_v2.py` handles this automatically with exponential back-off
(90s × attempt number). If 429s persist, increase `CALL_INTERVAL`:
```python
# Edit at top of run_llm_v2.py before running:
CALL_INTERVAL = 10.0   # increase from 6.0
```
Gemini API — do not use
The original `run_llm_calls.py` script used Gemini. It is deprecated. The free
tier was saturated. All scripts from `run_llm_v2.py` onwards use Groq only.
Colab session disconnect during SPSD run
The SPSD run does not checkpoint (each call is fast, checkpointing adds overhead).
If disconnected mid-run, re-run `run_spsd_v2.py` from the beginning — it overwrites
the output file. The model remains loaded if the Colab runtime is still active.
Model not found
```python
# Re-download:
MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')
# Or load from Drive:
MODEL_PATH = '/content/drive/MyDrive/spsd/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
spsd_v4.load_model(model_path=MODEL_PATH)
```
---
10. Architecture Summary
```
User prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Tier 1 — Rule-Based Gate  (~0 ms)              │
│  ├─ word count ≤ 15 → short_prompt              │
│  ├─ safety phrase match → safety_critical        │
│  ├─ medical keyword/regex → domain_medical       │  ──► PASSTHROUGH
│  └─ legal keyword/phrase → domain_legal          │
└─────────────────────────────────────────────────┘
    │ (passes)
    ▼
┌─────────────────────────────────────────────────┐
│  Complexity Scorer  (~0 ms)                     │
│  5 dimensions: social / semantic / structural / │
│  repetition / word_count → profile + ratio      │
│  structural_score ≥ 0.22 → structural passthru  │  ──► PASSTHROUGH
└─────────────────────────────────────────────────┘
    │ (passes)
    ▼
┌─────────────────────────────────────────────────┐
│  High-Fidelity Guard  (~0 ms)                   │
│  4 regex layers → extracts critical phrases     │
│  Seeds aux list (guaranteed in final packet)    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  SLM — Qwen2.5-1.5B Q4_K_M  (~80–180 ms CPU)  │
│  Input: [DOMAIN] tagged prompt + ratio target   │
│  Output: JSON {compressed_prompt, intent, tone, │
│                urgency, aux, confidence}         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Economic Gates  (~0 ms)                        │
│  ├─ Tier 1b: safety re-check on compressed out  │
│  ├─ Confidence gate: 0.65 / 0.72 / 0.80        │
│  └─ Net saving gate: ≥ 10 tokens net            │  ──► PASSTHROUGH (any fail)
└─────────────────────────────────────────────────┘
    │ (all pass)
    ▼
┌─────────────────────────────────────────────────┐
│  ALE — Adaptive Logic Engine                    │
│  Builds: D|tone|urgency|aux1;aux2;...           │
│          <compressed prompt>                    │
│  Static cached system prompt (~420 tokens)      │
└─────────────────────────────────────────────────┘
    │
    ▼
Frontier LLM (any provider — Groq/Anthropic/OpenAI)
```
Token economics
```
Net saving = tokens(original + 2) - tokens(D|header\ncompressed_prompt)
           = tokens(raw user turn) - tokens(distilled user turn)

ALE header overhead: ~6 tokens
Minimum net saving required: 10 tokens
System prompt: ~420 tokens — cached at 0.10× cost on Anthropic/OpenAI APIs
```
Key design invariants
Static system prompt always — any dynamic value in system prompt = cache miss
Score tone/urgency on original text — never on compressed output
HFG phrases are guaranteed in aux — SLM cannot compress them away
Tier 1b does NOT check word count — short compressed output is success
Medical and legal always passthrough — validated at 0 false positives
Net saving gate is sole economic arbiter — no compression ratio gate
Parse failure never silently becomes passthrough — confidence=0.0 fallback
W:<n> word budget removed — output length control via max_tokens at API level
---
SPSD v4.2 — Ajeet Kumar, Abhinit Sen — Indian School of Business, Hyderabad
