# SPSD v4.2 — Replication Guide

**Sentiment Preserving Semantic Distillation  + Adaptive Logic Engine**

This guide covers everything needed to replicate the full pipeline from scratch:
corpus fetch → SPSD distillation → LLM evaluation → hypothesis testing → paper.

\---

## Contents

1. [Overview](#1-overview)
2. [File Inventory](#2-file-inventory)
3. [Prerequisites](#3-prerequisites)
4. [Environment Setup](#4-environment-setup)
5. [Step-by-Step Replication](#5-step-by-step-replication)
6. [Configuration Reference](#6-configuration-reference)
7. [Frontier LLM Options](#7-frontier-llm-options)
8. [Expected Outputs](#8-expected-outputs)
9. [Troubleshooting](#9-troubleshooting)
10. [Architecture Summary](#10-architecture-summary)

\---

## 1\. Overview

SPSD is an on-device prompt compression pipeline. Before a user prompt reaches
a frontier LLM (GPT-4o, Llama-3, Claude, etc.), a 4-bit quantized 1.5B SLM
running on the edge device compresses it, stripping social scaffolding and
preserving semantic content. The frontier LLM receives a compact structured
packet and reconstructs a full natural-language response.

**Core claim (empirically validated):**

* Mean input token saving: 62.3 tokens per distilled call (t=16.17, p<0.001, d=1.92)
* Response quality: mean cosine similarity 0.763 vs 0.70 threshold (p<0.001)
* All 71 distilled calls produced positive savings (100%)

The pipeline has **no dependency on any specific frontier LLM provider**.
The evaluation used Groq (free tier). The `round\_trip()` demo function supports
Groq, Anthropic, and OpenAI.

\---

## 2\. File Inventory

### Core pipeline (required)

|File|Purpose|
|-|-|
|`spsd\_v4.py`|Main SPSD pipeline — Tier 1 gate, complexity scorer, HFG, SLM, economic gates|
|`ale\_prompt.py`|ALE packet builder — formats distilled output for frontier LLM|

### Corpus and evaluation scripts (run in order)

|File|Step|Input|Output|
|-|-|-|-|
|`fetch\_corpus\_v2.py`|1|HuggingFace API|`spsd\_corpus\_v2.csv`|
|`run\_spsd\_v2.py`|2|`spsd\_corpus\_v2.csv`|`spsd\_results\_v2.csv`|
|`run\_llm\_v2.py`|3|`spsd\_results\_v2.csv`|updates `spsd\_results\_v2.csv`|
|`score\_and\_test.py`|4|`spsd\_results\_v2.csv`|`spsd\_hypothesis\_v2.xlsx`|

### Legacy scripts (from earlier runs — not needed for fresh replication)

|File|Notes|
|-|-|
|`fetch\_support\_only.py`|Generates 32 synthetic support prompts (now included in corpus v2)|
|`run\_spsd\_only.py`|Older SPSD runner — use `run\_spsd\_v2.py` instead|
|`run\_llm\_support.py`|Older LLM runner — use `run\_llm\_v2.py` instead|
|`run\_llm\_topup.py`|Targeted re-run for missing rows — use `run\_llm\_v2.py` instead|
|`merge\_and\_score.py`|Older merger — use `score\_and\_test.py` instead|
|`score\_quality.py`|Standalone similarity scorer — now integrated into `score\_and\_test.py`|
|`run\_llm\_calls.py`|**Deprecated** — used Gemini (saturated). Do not use.|

### Results

|File|Contents|
|-|-|
|`spsd\_hypothesis\_final.xlsx`|Full hypothesis test results (5 sheets)|
|||

\---

## 3\. Prerequisites

### Accounts and API keys

|Service|Required for|Cost|Get key at|
|-|-|-|-|
|**Groq**|LLM evaluation calls|Free (rate-limited)|console.groq.com|
|**Google Colab**|Running the pipeline|Free tier sufficient|colab.research.google.com|
|HuggingFace|Corpus fetch (no auth needed)|Free, no account|—|
|Anthropic|Optional demo only|Paid|console.anthropic.com|
|OpenAI|Optional demo only|Paid|platform.openai.com|

The full replication pipeline uses only **Groq** (free tier) and **Google Colab**
(free tier). No paid API access is required.

### Groq free tier limits

|Limit|Value|
|-|-|
|Requests/day|14,400|
|Tokens/minute (llama-3.1-8b-instant)|\~20,000|
|Rate limit behaviour|HTTP 429, retry after 90s|

At 6 seconds between calls and 71 distilled prompts requiring 2 calls each
(raw + distilled), the full LLM evaluation takes approximately 15–20 minutes.

\---

## 4\. Environment Setup

### 4.1 Google Colab setup (recommended for full replication)

```python
# Cell 1 — Install llama-cpp-python (pre-built CPU wheel, \~2 min)
!pip install -q llama-cpp-python \\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Cell 2 — Install other dependencies
!pip install -q groq sentence-transformers huggingface\_hub \\
    scipy openpyxl matplotlib requests

# Cell 3 — Add Groq API key to Colab Secrets
# In Colab: click the key icon (🔑) in the left sidebar
# Add secret name: GROQ\_API\_KEY
# Value: your Groq API key from console.groq.com
```

### 4.2 Local setup (alternative)

```bash
# Python 3.10+ required
pip install llama-cpp-python --extra-index-url \\
    https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install groq sentence-transformers huggingface\_hub \\
    scipy openpyxl matplotlib requests

# Set environment variable
export GROQ\_API\_KEY="your\_groq\_key\_here"
```

### 4.3 Download the SLM model

```python
# Run once — downloads \~986 MB to /content/models/
# (In Colab, save to Google Drive to persist across sessions)
import sys
sys.path.insert(0, '/content')
import spsd\_v4
MODEL\_PATH = spsd\_v4.download\_model(dest\_dir='/content/models')
print(f"Model at: {MODEL\_PATH}")

# Optional: save to Drive so you don't re-download each session
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd/models', exist\_ok=True)
shutil.copy(MODEL\_PATH, '/content/drive/MyDrive/spsd/models/')
```

\---

## 5\. Step-by-Step Replication

Upload `spsd\_v4.py` and `ale\_prompt.py` to `/content/` before running any step.

### Critical: always reload modules after upload

```python
import sys, importlib

# Purge any cached versions
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules\[key]

import spsd\_v4, ale\_prompt
importlib.reload(spsd\_v4)
importlib.reload(ale\_prompt)
print(f"spsd\_v4 loaded   | SHORT\_LIMIT={spsd\_v4.SHORT\_PROMPT\_WORD\_LIMIT}")
print(f"ale\_prompt loaded | {len(ale\_prompt.ALE\_SYSTEM\_PROMPT.split())} words in system prompt")

# Verify v4.2 medical passthrough is active
test\_prompt = (
    "A 23-year-old pregnant woman at 22 weeks gestation presents to her physician "
    "with burning urination that started yesterday and she is concerned about "
    "treatment safety during pregnancy"
)
pt, reason = spsd\_v4.tier1\_check(test\_prompt)
assert pt and reason == "domain\_medical", \\
    f"Medical passthrough not working: pt={pt} reason={reason}"
print(f"Medical passthrough: OK (reason={reason})")

# Verify code domain is tag-only (not passthroughed)
code\_test = (
    "I keep getting a NullPointerException in my Java checkout service every time "
    "a user applies a discount code the stack trace shows CartService line 42"
)
pt2, reason2 = spsd\_v4.tier1\_check(code\_test)
assert not pt2, f"Code should NOT passthrough but got pt={pt2} reason={reason2}"
print(f"Code tag-only: OK (pt={pt2})")
```

\---

### Step 1 — Fetch corpus (\~5 min)

Upload `fetch\_corpus\_v2.py` to `/content/`, then:

```python
%run fetch\_corpus\_v2.py
```

**Output:** `/content/spsd\_corpus\_v2.csv` — 150 prompts across 7 categories.

**What it fetches:**

|Category|n|Source|
|-|-|-|
|verbose\_social|25|Synthetic (built into script)|
|general\_conversational|30|WildChat-4.8M (HuggingFace, no auth)|
|code\_technical|25|CodeFeedback-Filtered-Instruction (HuggingFace)|
|multi\_intent\_linked|25|WildChat-4.8M + UltraChat\_200k (HuggingFace)|
|single\_intent\_clear|15|Synthetic (built into script)|
|high\_stakes\_medical|15|MedQA-USMLE-4-options (HuggingFace)|
|short\_passthrough|15|Bitext customer support (HuggingFace)|

**If a HuggingFace dataset is unavailable**, the script will print a fetch error
and continue. Categories with fewer than their target count will be noted in the
summary. The pipeline tolerates a corpus of 130–150 prompts without issue.

\---

### Step 2 — Run SPSD distillation (\~30–45 min on Colab CPU)

Upload `run\_spsd\_v2.py` to `/content/`, then:

```python
%run run\_spsd\_v2.py
```

**Output:** `/content/spsd\_results\_v2.csv`

**What to expect:**

* Medical prompts → `PASSTHRU reason=domain\_medical`
* Short prompts → `PASSTHRU reason=short\_prompt`
* Distilled prompts → `DISTILL saving=+XXt (YYYms)`
* Typical saving range: +22 to +146 tokens
* Mean latency per SLM call: 80–180ms (CPU), 50–100ms (NPU)

**Save to Drive after this step:**

```python
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd', exist\_ok=True)
shutil.copy('/content/spsd\_results\_v2.csv',
            '/content/drive/MyDrive/spsd/spsd\_results\_v2.csv')
print("Saved")
```

\---

### Step 3 — Run LLM calls (\~15–20 min)

Upload `run\_llm\_v2.py` to `/content/`. Ensure `GROQ\_API\_KEY` is set in Colab
Secrets (left sidebar → 🔑 icon → add key). Then:

```python
%run run\_llm\_v2.py
```

**What it does:**

* Reads all `passthrough=False` rows from `spsd\_results\_v2.csv`
* For each row: makes a raw call (`P\\n<original>`) and a distilled call (ALE packet)
* Skips passthrough rows entirely — no LLM calls needed for those
* Saves a completion tracker `llm\_v2\_done.json` — safe to stop and restart
* Checkpoints every 10 rows

**If interrupted:** simply re-run `%run run\_llm\_v2.py` — it resumes from last checkpoint.

**To force a full re-run** (e.g., after changing the corpus):

```python
import os
for f in \['/content/llm\_v2\_done.json']:
    if os.path.exists(f): os.remove(f)
# Also clear old responses from spsd\_results\_v2.csv if needed:
import csv
with open('/content/spsd\_results\_v2.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row.get('passthrough') == 'False':
        for col in \['raw\_response','dist\_response','raw\_output\_tokens',
                    'dist\_output\_tokens','llm\_model','semantic\_similarity','quality\_flag']:
            row\[col] = ''
# Write back (add all\_keys logic as in the script)
```

**What to expect in output:**

```
\[ 1/71] P001 \[verbose\_social   ] raw=127 dist=116 save\_in=+77t \[OK ]
\[ 2/71] P002 \[verbose\_social   ] raw=134 dist=108 save\_in=+84t \[OK ]
...
```

`save\_in` is the input token saving (positive = SPSD saved tokens).
`raw` and `dist` are output token counts from the LLM (secondary metric, not in H1).

\---

### Step 4 — Score quality + hypothesis tests

Upload `score\_and\_test.py` to `/content/`, then:

```python
%run score\_and\_test.py
```

**What it does:**

1. Scores cosine similarity on all paired response rows using `all-MiniLM-L6-v2`
2. Writes similarity scores back to `spsd\_results\_v2.csv`
3. Runs T1 (token saving t-test), T2 (quality t-test), T3 (Kruskal-Wallis ANOVA)
4. Generates 4 charts
5. Saves `spsd\_hypothesis\_v2.xlsx` (5 sheets)

**Expected results:**

```
T1: t=16.17 p=1.23e-25 d=1.92 → REJECT H0 ✓
T2: t=4.22  p=0.000038 d=0.52 → REJECT H0 ✓
T3: H=4.96  p=0.175          → NOT significant (negative result)
```

\---

### Step 5 — Save everything to Drive

```python
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd', exist\_ok=True)

for fname in \['spsd\_corpus\_v2.csv', 'spsd\_results\_v2.csv',
              'spsd\_hypothesis\_v2.xlsx']:
    shutil.copy(f'/content/{fname}',
                f'/content/drive/MyDrive/spsd/{fname}')
    print(f"Saved: {fname}")
```

\---

## 6\. Configuration Reference

### spsd\_v4.py — key constants

|Constant|Value|Meaning|
|-|-|-|
|`SHORT\_PROMPT\_WORD\_LIMIT`|15|Prompts ≤15 words always passthrough|
|`ALE\_HEADER\_OVERHEAD\_TOKENS`|6|Token cost of D\|tone\|urgency\|aux header|
|`MIN\_NET\_TOKEN\_SAVING`|10|Minimum saving to justify distillation|
|`SLM\_TEMPERATURE`|0.1|SLM inference temperature (low = deterministic)|

### Dynamic confidence thresholds (tier1\_check)

|Prompt length|Confidence threshold|
|-|-|
|≤ 30 words|0.80|
|≤ 60 words|0.72|
|> 60 words|0.65|

### Passthrough triggers (in order)

1. `short\_prompt` — word count ≤ 15
2. `safety\_critical` — hard safety phrases (chest pain, overdose, 911...)
3. `domain\_medical` — 65 clinical keywords + clinical structure regex
4. `domain\_legal` — 30 legal keywords/phrases
5. `complexity\_structural\_density\_X.XX` — structural score ≥ 0.22 (code/math)
6. `low\_confidence\_X.XX\_threshold\_X.XX` — SLM confidence below threshold
7. `no\_token\_saving\_±Xt` — net saving below MIN\_NET\_TOKEN\_SAVING

### run\_llm\_v2.py — key constants

|Constant|Value|Meaning|
|-|-|-|
|`GROQ\_MODEL`|`llama-3.1-8b-instant`|Frontier model for evaluation|
|`MAX\_TOKENS`|150|Max output tokens per call|
|`TEMPERATURE`|0.4|LLM temperature|
|`CALL\_INTERVAL`|6.0s|Seconds between API calls (rate limit buffer)|
|`RETRY\_WAIT`|90s|Wait on HTTP 429 (rate limit)|
|`MAX\_RETRIES`|4|Max retries per call before recording error|

\---

## 7\. Frontier LLM Options

The evaluation pipeline uses **Groq** (free tier). The `round\_trip()` demo
function in `ale\_prompt.py` supports three providers. To use a different
frontier LLM in the evaluation, change `GROQ\_MODEL` in `run\_llm\_v2.py`
to any Groq-hosted model, or adapt the call\_groq function for your provider.

### Groq models (recommended for replication)

|Model|Notes|
|-|-|
|`llama-3.1-8b-instant`|Used in evaluation. Fast, high TPM limit.|
|`llama-3.3-70b-versatile`|Higher quality, lower TPM limit (\~6,000/min)|
|`mixtral-8x7b-32768`|Alternative, good quality|

### Anthropic (optional, paid)

```python
from ale\_prompt import round\_trip
result = round\_trip(
    prompt="Your prompt here",
    api\_key="YOUR\_ANTHROPIC\_KEY",
    model="claude-sonnet-4-6",   # or claude-opus-4-6, claude-haiku-4-5
    provider="anthropic"
)
result.display()
```

### OpenAI (optional, paid)

```python
result = round\_trip(
    prompt="Your prompt here",
    api\_key="YOUR\_OPENAI\_KEY",
    model="gpt-4o",
    provider="openai"
)
result.display()
```

### Using SPSD without any cloud LLM (distillation only)

```python
import sys
sys.path.insert(0, '/path/to/spsd')
import spsd\_v4

spsd\_v4.load\_model(model\_path='/path/to/qwen2.5-1.5b-instruct-q4\_k\_m.gguf')
result = spsd\_v4.distill("Your prompt here")
print(result.summary())
print(f"Token saving: {result.token\_saving}")
print(f"Passthrough: {result.passthrough} ({result.passthrough\_reason})")
```

\---

## 8\. Expected Outputs

### spsd\_corpus\_v2.csv

150 rows, 7 columns: `id, category, word\_count, prompt, source, intent\_label, domain\_tag`

### spsd\_results\_v2.csv

150 rows, 48 columns including:

* SPSD output: `passthrough, passthrough\_reason, complexity\_profile, token\_saving\_input, compression\_ratio, confidence, hfg\_aux, ale\_user\_turn`
* LLM responses: `raw\_response, dist\_response, raw\_output\_tokens, dist\_output\_tokens`
* Quality: `semantic\_similarity, quality\_flag`

### spsd\_hypothesis\_v2.xlsx

5 sheets:

1. **Executive Summary** — KPI cards + hypothesis table + key findings
2. **Token Saving** — T1 descriptive stats + by-category table + 2 charts
3. **Quality Analysis** — T2 stats + by-category table + exclusion note + 2 charts
4. **Full Data** — all 150 rows colour-coded by passthrough/quality/category
5. **Methodology** — corpus, model, tests, limitations

### Routing summary (approximate, varies by corpus fetch)

|Outcome|Count|%|
|-|-|-|
|Distilled|71|47.3%|
|Passthrough (all reasons)|79|52.7%|
|— domain\_medical|17|11.3%|
|— short\_prompt|19|12.7%|
|— no\_token\_saving|35|23.3%|
|— domain\_legal|3|2.0%|
|— other|5|3.3%|

\---

## 9\. Troubleshooting

### "MEDICAL passthrough not working" assertion fails on run\_spsd\_v2.py

The test prompt in the assertion is under 15 words. This is a test prompt issue,
not a pipeline issue. Patch directly in Colab:

```python
content = open("/content/run\_spsd\_v2.py").read()
content = content.replace(
    '"A 23-year-old pregnant woman at 22 weeks gestation presents with burning urination"',
    '"A 23-year-old pregnant woman at 22 weeks gestation presents to her physician '
    'with burning urination that started yesterday and she is concerned about '
    'treatment safety during pregnancy"'
)
open("/content/run\_spsd\_v2.py", "w").write(content)
%run run\_spsd\_v2.py
```

### Medical prompts still getting distilled despite fix

The old module is cached in memory. Run the full purge:

```python
import sys, importlib
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules\[key]
import spsd\_v4, ale\_prompt
importlib.reload(spsd\_v4)
importlib.reload(ale\_prompt)
```

Then verify: `spsd\_v4.tier1\_check("A patient presents to the physician with chest symptoms")` must return `(True, 'domain\_medical')`.

### LLM run resumes from wrong position

Delete the completion tracker:

```python
import os
if os.path.exists('/content/llm\_v2\_done.json'):
    os.remove('/content/llm\_v2\_done.json')
```

### Groq 429 rate limit

`run\_llm\_v2.py` handles this automatically with exponential back-off
(90s × attempt number). If 429s persist, increase `CALL\_INTERVAL`:

```python
# Edit at top of run\_llm\_v2.py before running:
CALL\_INTERVAL = 10.0   # increase from 6.0
```

### Gemini API — do not use

The original `run\_llm\_calls.py` script used Gemini. It is deprecated. The free
tier was saturated. All scripts from `run\_llm\_v2.py` onwards use Groq only.

### Colab session disconnect during SPSD run

The SPSD run does not checkpoint (each call is fast, checkpointing adds overhead).
If disconnected mid-run, re-run `run\_spsd\_v2.py` from the beginning — it overwrites
the output file. The model remains loaded if the Colab runtime is still active.

### Model not found

```python
# Re-download:
MODEL\_PATH = spsd\_v4.download\_model(dest\_dir='/content/models')
# Or load from Drive:
MODEL\_PATH = '/content/drive/MyDrive/spsd/models/qwen2.5-1.5b-instruct-q4\_k\_m.gguf'
spsd\_v4.load\_model(model\_path=MODEL\_PATH)
```

\---

## 10\. Architecture Summary

```
User prompt
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Tier 1 — Rule-Based Gate  (\~0 ms)              │
│  ├─ word count ≤ 15 → short\_prompt              │
│  ├─ safety phrase match → safety\_critical        │
│  ├─ medical keyword/regex → domain\_medical       │  ──► PASSTHROUGH
│  └─ legal keyword/phrase → domain\_legal          │
└─────────────────────────────────────────────────┘
    │ (passes)
    ▼
┌─────────────────────────────────────────────────┐
│  Complexity Scorer  (\~0 ms)                     │
│  5 dimensions: social / semantic / structural / │
│  repetition / word\_count → profile + ratio      │
│  structural\_score ≥ 0.22 → structural passthru  │  ──► PASSTHROUGH
└─────────────────────────────────────────────────┘
    │ (passes)
    ▼
┌─────────────────────────────────────────────────┐
│  High-Fidelity Guard  (\~0 ms)                   │
│  4 regex layers → extracts critical phrases     │
│  Seeds aux list (guaranteed in final packet)    │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  SLM — Qwen2.5-1.5B Q4\_K\_M  (\~80–180 ms CPU)  │
│  Input: \[DOMAIN] tagged prompt + ratio target   │
│  Output: JSON {compressed\_prompt, intent, tone, │
│                urgency, aux, confidence}         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Economic Gates  (\~0 ms)                        │
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
│  Static cached system prompt (\~420 tokens)      │
└─────────────────────────────────────────────────┘
    │
    ▼
Frontier LLM (any provider — Groq/Anthropic/OpenAI)
```

### Token economics

```
Net saving = tokens(original + 2) - tokens(D|header\\ncompressed\_prompt)
           = tokens(raw user turn) - tokens(distilled user turn)

ALE header overhead: \~6 tokens
Minimum net saving required: 10 tokens
System prompt: \~420 tokens — cached at 0.10× cost on Anthropic/OpenAI APIs
```

### Key design invariants

1. **Static system prompt always** — any dynamic value in system prompt = cache miss
2. **Score tone/urgency on original text** — never on compressed output
3. **HFG phrases are guaranteed in aux** — SLM cannot compress them away
4. **Tier 1b does NOT check word count** — short compressed output is success
5. **Medical and legal always passthrough** — validated at 0 false positives
6. **Net saving gate is sole economic arbiter** — no compression ratio gate
7. **Parse failure never silently becomes passthrough** — confidence=0.0 fallback
8. **W:<n> word budget removed** — output length control via max\_tokens at API level

\---

*SPSD v4.2 — Ajeet Kumar, Abhinit Sen — Indian School of Business, Hyderabad*

