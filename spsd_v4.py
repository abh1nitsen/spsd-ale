"""
SPSD v4 — Semantic Prompt Structural Distillation
SLM-based single-pass distillation pipeline.

Architecture:
    User prompt
        ↓
    Tier 1  — rule-based guards (~0ms, zero model cost)
        ↓ if not passthrough
    SLM     — Qwen2.5-1.5B-Instruct Q4_K_M via llama-cpp-python
              single call → compressed_prompt + metadata envelope
        ↓
    Tier 1b — fallback safety re-check on compressed_prompt (~0ms)
        ↓
    ALE     — builds frontier LLM packet

Design constraints:
    - SLM must run on CPU, target <200ms on free-tier Colab CPU
    - No frontier model dependency in distillation layer
    - Passthrough is not failure — it is the system working correctly
    - Urgency/tone scored on original text, never distilled payload
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

MODEL_REPO   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE   = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH   = Path("./models") / MODEL_FILE   # local cache path

# Inference parameters — tuned for latency on free-tier Colab CPU
SLM_CONTEXT_LENGTH   = 2048
SLM_MAX_TOKENS       = 300    # envelope is compact
SLM_TEMPERATURE      = 0.1    # near-deterministic for structured output
SLM_TOP_P            = 0.9
SLM_N_THREADS        = 2      # free-tier Colab: 2 CPU cores safe default

# ─────────────────────────────────────────────
# TOKEN ECONOMICS CONFIGURATION
# ─────────────────────────────────────────────

# Short prompt guard — prompts at or under this word count passthrough.
# Raised from 5 to 15: anything shorter than ~15 words cannot produce
# net-positive token saving once ALE header overhead (~35 tokens) is added.
SHORT_PROMPT_WORD_LIMIT = 15

# ALE header overhead in tokens — compact format:
# "D|tone|urgency|W:n|" = ~8 tokens fixed.
# Passthrough format "P\n" = ~2 tokens (negligible, passthrough skips saving gate).
# Aux items encoded inline as semicolon-separated — ~2 tokens per item (vs 4 before).
ALE_HEADER_OVERHEAD_TOKENS = 8

# Minimum net token saving to justify distillation.
# Set to 10 — a saving of fewer than 10 tokens is noise, not a win.
MIN_NET_TOKEN_SAVING = 10

# Dynamic confidence threshold — varies by prompt length.
# Short prompts: higher bar (less room for error, smaller saving upside).
# Long prompts: standard bar (more room to compress faithfully).
def dynamic_confidence_threshold(word_count: int) -> float:
    """
    word_count 16-30 : 0.80 — short prompts, need high confidence
    word_count 31-60 : 0.72 — medium prompts, moderate bar
    word_count > 60  : 0.65 — long prompts, standard bar
    """
    if word_count <= 30:
        return 0.80
    elif word_count <= 60:
        return 0.72
    else:
        return 0.65

# ─────────────────────────────────────────────
# TIER 1 — DOMAIN KEYWORD SETS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# TIER 1 — DOMAIN DETECTION (tag, not block)
# ─────────────────────────────────────────────
# Tier 1 no longer passthroughs on domain.
# It detects domain and attaches a marker so the SLM
# can compress with domain awareness.
# Only TRUE passthrough cases: prompts so short distillation
# cannot be net-positive regardless of compression quality.
# ─────────────────────────────────────────────

# Hard passthrough — prompts containing these should never be
# compressed because exact wording is safety-critical and
# any loss of fidelity could cause real harm.
# Keep this list SHORT and unambiguous.
HARD_PASSTHROUGH_PHRASES = {
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "heart attack", "overdose", "suicidal", "suicide", "self harm",
    "bleeding heavily", "unconscious", "not breathing", "anaphylaxis",
    "911", "999", "112",   # emergency numbers
}

# Domain detection — used for tagging only, not blocking
MEDICAL_KEYWORDS = {
    "diagnosis", "symptom", "prescription", "medication", "dosage",
    "surgery", "cancer", "tumor", "diabetes", "insulin", "chemotherapy",
    "vaccine", "allergy", "seizure", "stroke", "cardiac", "hypertension",
    "depression", "anxiety", "schizophrenia", "bipolar", "adhd", "autism",
    "therapy", "psychiatrist", "psychologist", "antidepressant", "opioid",
    "narcotic", "painkiller", "ibuprofen", "paracetamol", "antibiotic",
    "biopsy", "mri", "ct scan", "pathology", "prognosis", "palliative",
}

LEGAL_KEYWORDS = {
    "lawsuit", "litigation", "attorney", "lawyer", "legal advice",
    "court", "verdict", "plaintiff", "defendant", "subpoena",
    "arbitration", "settlement", "contract", "liability", "negligence",
    "malpractice", "patent", "trademark", "copyright infringement",
    "gdpr", "statute", "jurisdiction", "appeal", "injunction",
    "divorce", "custody", "bankruptcy", "foreclosure", "eviction",
    "sue", "suing", "landlord", "tenant rights", "wrongful",
    "discrimination", "harassment", "employment law", "unfair dismissal",
    "small claims", "restraining order", "legal rights", "breach of contract",
}

_FINANCIAL_STRONG = {
    "tax", "mortgage", "401k", "ira", "roth", "dividend",
    "capital gains", "depreciation", "portfolio", "hedge fund",
    "mutual fund", "etf", "bond", "bonds", "equity", "derivative",
    "futures", "forex", "cryptocurrency", "bitcoin", "ethereum",
    "annuity", "pension", "fiduciary", "brokerage", "securities",
    "index fund", "index funds", "stock market", "shares", "interest rate",
    "compound interest", "net worth", "asset allocation", "retirement fund",
}

_FINANCIAL_WEAK = {
    "account", "bank", "pay", "payment", "bill", "charge", "refund",
    "credit", "fee", "money", "cash", "cost", "price", "budget",
    "expense", "transfer", "deposit", "withdraw", "balance", "invoice",
}

_FINANCIAL_ADVICE_VERBS = {
    "invest", "advise", "recommend", "allocate", "hedge",
    "speculate", "trade", "rebalance", "diversify", "claim", "deduct",
}

# Code — strong unambiguous terms only for tagging
CODE_STRONG = {
    "python", "javascript", "typescript", "kotlin", "swift", "golang",
    "haskell", "scala", "php", "bash", "shell", "regex", "yaml",
    "kubernetes", "webpack", "npm", "pip", "conda", "recursion",
    "algorithm", "compile", "compiler", "debugger", "async", "await",
    "lambda", "decorator", "polymorphism", "refactor", "linter",
    "undefined", "boolean", "tuple", "iterator", "generator",
    "middleware", "microservice", "jwt", "oauth", "webhook", "graphql",
    "virtualenv", "pytorch", "tensorflow", "pandas", "numpy",
}

CODE_PHRASES = {
    "syntax error", "null pointer", "stack overflow", "time complexity",
    "pull request", "code review", "type error", "index out of bounds",
    "memory leak", "race condition", "deadlock", "object oriented",
    "functional programming", "version control", "dependency injection",
}

# ─────────────────────────────────────────────
# POLITENESS MARKERS
# Used by SLM prompt builder to preserve tone signal.
# ─────────────────────────────────────────────

POLITENESS_MARKERS = {
    "sorry", "apologise", "apologize", "apologies", "forgive",
    "embarrassed", "ashamed", "bother you", "bothering you",
    "hate to ask", "feel bad", "incredibly sorry", "so sorry",
    "really sorry", "truly sorry", "hate to trouble", "please forgive",
}


def _detect_domain(text: str) -> Optional[str]:
    """
    Detect domain of prompt for tagging.
    Returns domain string or None.
    Priority: medical > legal > financial > code > None
    """
    text_lower = text.lower()

    if _word_boundary_match(text, MEDICAL_KEYWORDS):
        return "MEDICAL"

    if _phrase_match(text, LEGAL_KEYWORDS) or _word_boundary_match(text, LEGAL_KEYWORDS):
        return "LEGAL"

    if _word_boundary_match(text, _FINANCIAL_STRONG):
        return "FINANCIAL"
    has_weak_fin = _word_boundary_match(text, _FINANCIAL_WEAK)
    has_advice   = _word_boundary_match(text, _FINANCIAL_ADVICE_VERBS)
    if has_weak_fin and has_advice:
        return "FINANCIAL"

    if _word_boundary_match(text, CODE_STRONG) or _phrase_match(text, CODE_PHRASES):
        return "CODE"

    return None


def tier1_check(text: str) -> tuple[bool, Optional[str]]:
    """
    Minimal passthrough gate — only fires on:
    1. Very short prompts (net saving impossible regardless)
    2. Hard safety phrases (exact wording is safety-critical)

    All domain prompts (medical, legal, financial, code) are tagged
    and sent to the SLM — NOT blocked here.

    Returns (passthrough: bool, reason: str | None)
    """
    tokens = text.split()

    if len(tokens) <= SHORT_PROMPT_WORD_LIMIT:
        return True, "short_prompt"

    if _phrase_match(text, HARD_PASSTHROUGH_PHRASES):
        return True, "safety_critical"

    return False, None


def tier1b_check(compressed: str) -> tuple[bool, Optional[str]]:
    """
    Fallback safety re-check on SLM compressed output.
    Only checks hard safety phrases — NOT word count.
    A short compressed output is a success, not a failure.
    The original prompt already passed the word count gate at Tier 1.
    """
    if _phrase_match(compressed, HARD_PASSTHROUGH_PHRASES):
        return True, "safety_critical"
    return False, None

# ─────────────────────────────────────────────
# AUX URGENCY ESCALATORS
# ─────────────────────────────────────────────

AUX_URGENCY_ESCALATORS = {
    "crying", "screaming", "emergency", "accident", "deadline",
    "dying", "urgent", "asap", "right now", "immediately",
    "meeting in", "presentation in", "flight in", "interview in",
    "surgery", "hospital", "ambulance", "bleeding", "burning", "fire",
}


# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES FOR SLM PROMPT
# Cover: support/tracking, domain-tagged medical, domain-tagged code,
#        explanatory/idiom, multi-question, casual
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — Support / tracking (anxious, high urgency, active stressor)
INPUT: "Hi I'm so sorry to bother you, I know you must be really busy, but I ordered something yesterday and I can't find the tracking number anywhere in my emails and my kids are crying in the background and I really need to know where this package is. The order number is TRK-00492. I'm incredibly stressed."
OUTPUT:
{
  "compressed_prompt": "Can't find tracking number for order TRK-00492 placed yesterday. Checked email, not found.",
  "intent": "retrieval",
  "tone": "anxious/apologetic",
  "urgency": "high",
  "aux": ["kids crying", "order placed yesterday", "checked email already"],
  "confidence": 0.91
}

EXAMPLE 2 — Medical domain (tagged — compress but preserve clinical precision)
INPUT: "[MEDICAL] I've been taking ibuprofen 400mg twice a day for about two weeks for back pain and I'm wondering if that's too long and whether I should be worried about side effects given I also have mild hypertension."
OUTPUT:
{
  "compressed_prompt": "Is 400mg ibuprofen twice daily for 2 weeks too long, and are side effects a concern with mild hypertension?",
  "intent": "information",
  "tone": "neutral",
  "urgency": "medium",
  "aux": ["back pain context", "concurrent hypertension", "2-week duration"],
  "confidence": 0.87
}

EXAMPLE 3 — Code domain (tagged — compress but keep technical precision)
INPUT: "[CODE] I'm getting a null pointer exception in my Java checkout service on line 42 every time a user tries to apply a discount code. The stack trace shows it's in the CartService.applyDiscount method. I've checked that the discount object isn't null but it keeps throwing."
OUTPUT:
{
  "compressed_prompt": "NullPointerException in Java CartService.applyDiscount (line 42) when applying discount codes. Discount object confirmed non-null.",
  "intent": "action",
  "tone": "neutral",
  "urgency": "medium",
  "aux": ["checkout service", "triggered by discount code application", "discount object already checked"],
  "confidence": 0.92
}

EXAMPLE 4 — Multi-question (linked questions)
INPUT: "Two things: first, what's the difference between supervised and unsupervised learning? And based on that, which one would be better for clustering customer purchase data where I don't have predefined labels?"
OUTPUT:
{
  "compressed_prompt": "What is the difference between supervised and unsupervised learning? Given that, which suits clustering unlabelled customer purchase data?",
  "intent": "information",
  "tone": "neutral",
  "urgency": "low",
  "aux": ["questions are linked", "use case: customer purchase clustering", "no predefined labels"],
  "confidence": 0.90
}

EXAMPLE 5 — Casual / conversational
INPUT: "Hey, random question — do you think remote work is actually better for productivity or is it just something people say? I've been going back and forth on this."
OUTPUT:
{
  "compressed_prompt": "Is remote work genuinely better for productivity, or is it overstated?",
  "intent": "information",
  "tone": "casual",
  "urgency": "low",
  "aux": ["user is undecided", "wants reasoned view not just validation"],
  "confidence": 0.85
}
""".strip()

# ─────────────────────────────────────────────
# SLM SYSTEM PROMPT
# ─────────────────────────────────────────────

SLM_SYSTEM_PROMPT = """You are a prompt distillation engine. Compress user messages into shorter versions that preserve full intent, tone, urgency, and key context. Remove only filler words, repetition, and irrelevant narrative.

Some inputs are prefixed with a domain tag: [MEDICAL], [LEGAL], [FINANCIAL], [CODE].
For tagged inputs: compress aggressively but preserve domain-critical precision.
  [MEDICAL] — keep drug names, dosages, durations, conditions exactly as stated
  [LEGAL]   — keep all party names, dates, specific legal terms exactly
  [FINANCIAL] — keep all figures, instrument names, account types exactly
  [CODE]    — keep error messages, method names, line numbers, language exactly

Output ONLY a valid JSON object with these exact fields:
- compressed_prompt: shorter natural language version. Preserve intent, identifiers, emotional register. Never invent information.
- intent: one of [retrieval, action, information, social]
- tone: one or more of [anxious/apologetic, polite, casual, negative, neutral] joined with /
- urgency: one of [high, medium, low]
- aux: list of contextual details useful for answering (stressors, prior attempts, background, constraints). Empty list [] if none.
- confidence: float 0.0–1.0 for how faithfully you captured original intent

Rules:
1. Output ONLY the JSON. No preamble, no explanation, no markdown fences.
2. compressed_prompt must be natural language — not a schema or bullet list.
3. Never drop identifiers: order numbers, tracking IDs, URLs, drug names, error messages.
4. Never invent context not present in the original.
5. Compress hard — target 30-50% of original word count. Filler, politeness, and narrative are always removable.
6. If the message is genuinely too ambiguous to compress faithfully, set confidence below 0.60."""

# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class DistillResult:
    original_prompt:    str
    compressed_prompt:  str
    intent:             str
    tone:               str
    urgency:            str
    aux:                list[str]
    passthrough:        bool
    passthrough_reason: Optional[str]
    confidence:         float
    latency_ms:         float
    tier:               str             # "tier1", "tier1b", "slm"
    schema_valid:       bool = True
    token_saving:       int  = 0    # net tokens saved vs sending original
    domain:             Optional[str] = None  # MEDICAL/LEGAL/FINANCIAL/CODE or None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.passthrough:
            return (
                f"[PASSTHROUGH — {self.passthrough_reason}]\n"
                f"Original forwarded to frontier LLM unchanged.\n"
                f"Tier: {self.tier} | Latency: {self.latency_ms:.1f}ms"
            )
        domain_str = f" | domain={self.domain}" if self.domain else ""
        return (
            f"[DISTILLED — {self.tier} | {self.latency_ms:.1f}ms | "
            f"confidence={self.confidence:.2f} | "
            f"token_saving={self.token_saving:+d}{domain_str}]\n"
            f"Compressed : {self.compressed_prompt}\n"
            f"Intent     : {self.intent}\n"
            f"Tone       : {self.tone}\n"
            f"Urgency    : {self.urgency}\n"
            f"Aux        : {self.aux}"
        )


# ─────────────────────────────────────────────
# TIER 1 — RULE-BASED GUARDS
# ─────────────────────────────────────────────

def _word_boundary_match(text: str, word_set: set[str]) -> bool:
    """Match whole words only — prevents 'import' matching 'important'."""
    text_lower = text.lower()
    for word in word_set:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def _phrase_match(text: str, phrase_set: set[str]) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrase_set)


def _extract_identifiers(text: str) -> list[str]:
    """Extract reference numbers, URLs, order IDs."""
    patterns = [
        r'\b[A-Z]{2,6}-\d{3,10}\b',          # TRK-00492, ORD-123456
        r'\b\d{6,15}\b',                        # plain numeric order IDs
        r'https?://[^\s]+',                     # URLs
        r'\b[A-Z0-9]{8,20}\b',                  # alphanumeric reference codes
    ]
    identifiers = []
    for pattern in patterns:
        identifiers.extend(re.findall(pattern, text))
    return list(set(identifiers))


def tier1_check(text: str) -> tuple[bool, Optional[str]]:
    """
    Returns (passthrough: bool, reason: str | None).
    True = should passthrough, do not distill.
    """
    tokens = text.split()

    # Short prompt guard — raised to SHORT_PROMPT_WORD_LIMIT.
    # Prompts this short cannot produce net-positive token saving
    # once ALE header overhead is added, regardless of compression quality.
    if len(tokens) <= SHORT_PROMPT_WORD_LIMIT:
        return True, "short_prompt"

    text_lower = text.lower()

    if _phrase_match(text, HARD_PASSTHROUGH_PHRASES):
        return True, "safety_critical"

    return False, None


# ─────────────────────────────────────────────
# AUX URGENCY ESCALATION
# ─────────────────────────────────────────────

def _escalate_urgency(urgency: str, aux: list[str]) -> str:
    """
    If aux contains active stressors, promote urgency to high
    regardless of SLM-scored urgency value.
    Urgency is scored on ORIGINAL text (via SLM), so this is a
    belt-and-suspenders check on extracted aux phrases.
    """
    combined = " ".join(aux).lower()
    for escalator in AUX_URGENCY_ESCALATORS:
        if escalator in combined:
            return "high"
    return urgency


# ─────────────────────────────────────────────
# SLM INFERENCE
# ─────────────────────────────────────────────

_llm = None  # module-level singleton — load once


def load_model(model_path: Optional[str] = None) -> None:
    """
    Load Qwen2.5-1.5B-Instruct Q4_K_M.
    Call once at startup. Safe to call multiple times (no-op if loaded).
    
    model_path: override default path (useful for Colab where model
                is downloaded to a specific location).
    """
    global _llm
    if _llm is not None:
        return

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. "
            "Run: pip install llama-cpp-python"
        )

    path = model_path or str(MODEL_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run download_model() first or pass the correct path."
        )

    print(f"Loading SLM from {path} ...")
    t0 = time.time()
    _llm = Llama(
        model_path=path,
        n_ctx=SLM_CONTEXT_LENGTH,
        n_threads=SLM_N_THREADS,
        n_gpu_layers=0,         # CPU only
        verbose=False,
    )
    print(f"SLM loaded in {(time.time()-t0)*1000:.0f}ms")


def download_model(dest_dir: str = "./models") -> str:
    """
    Download Qwen2.5-1.5B-Instruct Q4_K_M from HuggingFace Hub.
    Returns path to downloaded file.
    Requires: pip install huggingface-hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub not installed. "
            "Run: pip install huggingface-hub"
        )

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO} ...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=dest_dir,
    )
    print(f"Downloaded to {path}")
    return path


def _build_slm_prompt(user_text: str, domain: Optional[str] = None) -> str:
    """Build the full prompt string for the SLM.
    Prepends domain tag if detected so the SLM preserves domain precision.
    """
    tagged = f"[{domain}] {user_text}" if domain else user_text
    return (
        f"<|im_start|>system\n{SLM_SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Here are examples of correct distillation:\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Now distill this prompt:\n"
        f"INPUT: \"{tagged}\"\n"
        f"OUTPUT:\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_slm_output(raw: str, original: str) -> dict:
    """
    Parse JSON from SLM output.
    Robust to: leading/trailing whitespace, partial markdown fences,
    extra text before/after the JSON object.
    Returns parsed dict or fallback with confidence=0.
    """
    # Strip markdown fences if model added them
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()

    # Find the JSON object — from first { to last }
    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start == -1 or end == -1:
        return _fallback_envelope(original, "json_not_found")

    json_str = cleaned[start:end+1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to salvage by fixing common issues (trailing commas)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return _fallback_envelope(original, "json_parse_error")

    return parsed


def _fallback_envelope(original: str, reason: str) -> dict:
    """
    Parse failure must NEVER silently become passthrough.
    Return a low-confidence envelope — visible, diagnosable.
    """
    return {
        "compressed_prompt": original,
        "intent": "unknown",
        "tone": "neutral",
        "urgency": "medium",
        "aux": [],
        "passthrough": False,
        "confidence": 0.0,
        "_parse_failure": reason,
    }


def _validate_envelope(env: dict) -> tuple[dict, bool]:
    """
    Validate required fields and types.
    Fill missing fields with safe defaults.
    Returns (envelope, is_valid).
    """
    required = ["compressed_prompt", "intent", "tone", "urgency", "aux", "confidence"]
    valid = True

    defaults = {
        "compressed_prompt": "",
        "intent": "unknown",
        "tone": "neutral",
        "urgency": "medium",
        "aux": [],
        "confidence": 0.0,
    }

    for key in required:
        if key not in env:
            env[key] = defaults[key]
            valid = False

    # Type coercions
    if not isinstance(env["aux"], list):
        env["aux"] = [str(env["aux"])] if env["aux"] else []
        valid = False

    try:
        env["confidence"] = float(env["confidence"])
    except (ValueError, TypeError):
        env["confidence"] = 0.0
        valid = False

    # Clamp confidence
    env["confidence"] = max(0.0, min(1.0, env["confidence"]))

    # Compressed prompt must not be empty
    if not env.get("compressed_prompt", "").strip():
        valid = False

    return env, valid


def _run_slm(text: str, domain: Optional[str] = None) -> tuple[dict, float]:
    """
    Run SLM inference on text.
    Returns (envelope dict, latency_ms).
    Raises RuntimeError if model not loaded.
    """
    if _llm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    prompt = _build_slm_prompt(text, domain=domain)

    t0 = time.time()
    response = _llm(
        prompt,
        max_tokens=SLM_MAX_TOKENS,
        temperature=SLM_TEMPERATURE,
        top_p=SLM_TOP_P,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )
    latency_ms = (time.time() - t0) * 1000

    raw = response["choices"][0]["text"].strip()
    envelope = _parse_slm_output(raw, text)
    envelope, _ = _validate_envelope(envelope)

    return envelope, latency_ms


# ─────────────────────────────────────────────
# TOKEN ECONOMICS HELPERS
# ─────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Lightweight token estimator — no tokenizer dependency.
    Approximation: 1 token ≈ 0.75 words (standard rule of thumb).
    Good enough for the net saving gate; real tokenizers cost latency.
    """
    return max(1, round(len(text.split()) / 0.75))


def _compute_net_saving(original: str, compressed: str, aux: list) -> int:
    """
    Net token saving = tokens(original) - tokens(distilled packet)

    Distilled packet cost:
        tokens(compressed_prompt)
      + ALE_HEADER_OVERHEAD_TOKENS (fixed HEADER block)
      + aux overhead (~4 tokens per aux item for the "  - item" lines)

    A positive value means distillation saves tokens.
    A negative value means distillation costs more than passthrough.
    """
    original_tokens    = _estimate_tokens(original)
    compressed_tokens  = _estimate_tokens(compressed)
    aux_overhead       = len(aux) * 2   # inline semicolon-separated: ~2 tokens per item
    packet_tokens      = compressed_tokens + ALE_HEADER_OVERHEAD_TOKENS + aux_overhead
    return original_tokens - packet_tokens


# ─────────────────────────────────────────────
# MAIN DISTILL ENTRY POINT
# ─────────────────────────────────────────────

def distill(text: str, model_path: Optional[str] = None) -> DistillResult:
    """
    Main entry point. Returns DistillResult.

    text:       raw user prompt
    model_path: override model path (for Colab or custom setups)
    """
    t_start = time.time()
    text = text.strip()

    # ── Tier 1 ──────────────────────────────
    passthrough, reason = tier1_check(text)
    if passthrough:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=reason,
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="tier1",
            domain=None,
        )

    # ── Domain detection ─────────────────────
    # Tag the domain for the SLM — does not block, only informs compression
    domain = _detect_domain(text)

    # ── SLM ─────────────────────────────────
    # Lazy load if not already loaded
    if _llm is None:
        load_model(model_path)

    try:
        envelope, slm_latency_ms = _run_slm(text, domain=domain)
    except Exception as e:
        # SLM failure — safe fallback to passthrough with reason
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"slm_error: {str(e)}",
            confidence=0.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm_error",
            domain=domain,
        )

    # ── Tier 1b — fallback safety re-check ──
    compressed = envelope.get("compressed_prompt", text)
    tier1b_flag, tier1b_reason = tier1b_check(compressed)
    if tier1b_flag:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"tier1b_{tier1b_reason}",
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="tier1b",
            domain=domain,
        )

    # ── Dynamic confidence gate ──────────────
    # Threshold scales with prompt length — short prompts need higher
    # confidence because the margin for error is smaller and the
    # compression upside is lower.
    word_count  = len(text.split())
    threshold   = dynamic_confidence_threshold(word_count)
    confidence  = envelope.get("confidence", 0.0)
    if confidence < threshold:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"low_confidence_{confidence:.2f}_threshold_{threshold:.2f}",
            confidence=confidence,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm",
            token_saving=0,
            domain=domain,
        )

    # ── Net token saving gate ────────────────
    # Single economic gate — if the full packet (compressed + header +
    # aux overhead) does not save at least MIN_NET_TOKEN_SAVING tokens
    # vs sending the original, passthrough. This catches poor compressions,
    # marginal compressions, and prompts too short to benefit — without
    # a separate ratio gate that penalises the SLM for keeping useful context.
    aux       = envelope.get("aux", [])
    net_saving = _compute_net_saving(text, compressed, aux)
    if net_saving < MIN_NET_TOKEN_SAVING:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"no_token_saving_{net_saving:+d}_tokens",
            confidence=confidence,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm",
            token_saving=net_saving,
            domain=domain,
        )

    # ── Aux urgency escalation ───────────────
    urgency  = _escalate_urgency(envelope.get("urgency", "medium"), aux)

    # ── Identifier preservation check ───────
    # Any identifiers in the original must appear in compressed_prompt
    identifiers = _extract_identifiers(text)
    for ident in identifiers:
        if ident not in compressed:
            compressed += f" [{ident}]"

    total_latency_ms = (time.time() - t_start) * 1000

    return DistillResult(
        original_prompt=text,
        compressed_prompt=compressed,
        intent=envelope.get("intent", "unknown"),
        tone=envelope.get("tone", "neutral"),
        urgency=urgency,
        aux=aux,
        passthrough=False,
        passthrough_reason=None,
        confidence=confidence,
        latency_ms=total_latency_ms,
        tier="slm",
        schema_valid=True,
        token_saving=net_saving,
        domain=domain,
    )


# ─────────────────────────────────────────────
# BATCH EVALUATION
# ─────────────────────────────────────────────

def run_batch(prompts: list[str], model_path: Optional[str] = None) -> dict:
    """
    Run distillation on a list of prompts.
    Returns summary stats + per-prompt results.
    Useful for Colab evaluation cells.
    """
    results = []
    passthrough_count = 0
    tier1_count  = 0
    tier1b_count = 0
    slm_count    = 0
    latencies    = []
    total_tokens_saved = 0

    for prompt in prompts:
        r = distill(prompt, model_path=model_path)
        results.append(r)
        latencies.append(r.latency_ms)
        total_tokens_saved += r.token_saving

        if r.passthrough:
            passthrough_count += 1
            if r.tier == "tier1":
                tier1_count += 1
            elif r.tier == "tier1b":
                tier1b_count += 1
        else:
            slm_count += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    summary = {
        "total": len(prompts),
        "passthrough": passthrough_count,
        "distilled": slm_count,
        "tier1_exits": tier1_count,
        "tier1b_exits": tier1b_count,
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "passthrough_rate": f"{passthrough_count/len(prompts)*100:.1f}%",
        "total_tokens_saved": total_tokens_saved,
        "results": results,
    }
    return summary


def print_batch_summary(summary: dict) -> None:
    print("\n" + "="*50)
    print("BATCH EVALUATION SUMMARY")
    print("="*50)
    print(f"Total prompts    : {summary['total']}")
    print(f"Distilled        : {summary['distilled']}")
    print(f"Passthrough      : {summary['passthrough']} ({summary['passthrough_rate']})")
    print(f"  ↳ Tier 1 exits : {summary['tier1_exits']}")
    print(f"  ↳ Tier 1b exits: {summary['tier1b_exits']}")
    print(f"Total tokens saved: {summary['total_tokens_saved']:+d}")
    print(f"Avg latency      : {summary['avg_latency_ms']}ms")
    print(f"P95 latency      : {summary['p95_latency_ms']}ms")
    print("="*50)
    for i, r in enumerate(summary["results"]):
        print(f"\n[{i+1}] {r.summary()}")


# ─────────────────────────────────────────────
# INTERACTIVE REPL
# ─────────────────────────────────────────────

def run_interactive(model_path: Optional[str] = None) -> None:
    """
    Interactive REPL for manual testing.
    Type 'quit' or 'exit' to stop.
    Model loads lazily on first non-passthrough prompt.
    """
    print("SPSD v4 — Interactive REPL")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        result = distill(text, model_path=model_path)
        print("\n" + result.summary() + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_interactive()
