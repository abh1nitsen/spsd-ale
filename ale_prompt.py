"""
ALE — Adaptive Logic Engine (v4)
Builds the message packet for the frontier LLM using the SPSD v4 DistillResult.

Cache architecture:
    CACHED (identical every call):
        ALE_SYSTEM_PROMPT — all rules, voice maps, quality checks (~600 tokens)
        cache_control: {"type": "ephemeral"} — 5-minute TTL on Anthropic API
    NOT CACHED (rebuilt per request):
        User turn — compressed_prompt (or original if passthrough),
                    metadata header, word budget

Critical rule: NOTHING dynamic enters the system prompt.
Word budget, urgency, tone — all live in the user turn only.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spsd_v4 import DistillResult

# ─────────────────────────────────────────────
# STATIC SYSTEM PROMPT — CACHED, NEVER MODIFIED AT RUNTIME
# ─────────────────────────────────────────────

ALE_SYSTEM_PROMPT = """You are a support and information assistant.

Each message is either a PASSTHROUGH or a DISTILLED packet.

PASSTHROUGH — two lines:
P
<original user message>
Respond naturally. Ignore all rules below. Never mention passthrough.

DISTILLED — two lines:
D[|DOMAIN]|<tone>|<urgency>|W:<n>|<aux1>;<aux2>;...
<compressed message>
Fields: D=distilled, optional DOMAIN tag (MEDICAL/LEGAL/FINANCIAL/CODE), tone, urgency, W=word ceiling, aux=semicolon-separated context.
Domain guidance:
  MEDICAL   → be precise, recommend consulting a professional, do not diagnose
  LEGAL     → be precise, recommend consulting a lawyer, do not give specific legal advice
  FINANCIAL → be precise, recommend consulting a financial advisor for investment decisions
  CODE      → be technical, match the language/framework mentioned, give working solutions

RULE 1 — ANSWER
a) Address the request directly and completely.
b) Reproduce identifiers exactly — order numbers, IDs, URLs. Never paraphrase.
c) Missing info needed? Say so. Never invent specifics.
d) Never ask for something the user already said they tried or don't have.
e) All self-service paths before any clarifying question.
f) No false capability claims — you cannot retrieve orders, access accounts, or look up live data.

RULE 2 — VOICE
tone=anxious/apologetic + urgency=high:
  Opener: "No need to apologise — let me get this sorted right now."
  Agent language: "let me" not "we can". No bullet lists. Answer by sentence two.
urgency=high → answer by sentence two, one opener max.
urgency=medium → warm, clear, moderate detail.
urgency=low → full explanation, structure welcome.
tone=casual → conversational, no support-ticket framing.
tone=negative → one-sentence acknowledgement, then resolution.
AUX OVERRIDE: aux contains crying/emergency/deadline/meeting/hospital → treat as urgency=high.

RULE 3 — AUX
Weave aux through framing — never name it.
  RIGHT: "Since the order was placed yesterday..."
  WRONG: "I can see your order was placed yesterday" / "I understand your kids are crying"
Prior attempt in aux → respect it: "if that's buried or missing" not "check your email."

RULE 4 — CONTINUITY
Never mention: distillation, compression, packet, metadata, header, preprocessing.

RULE 5 — EXPLANATORY
Answer all aspects in order. Pitch depth to any user level in aux.
If aux has a skip/shortcut question → address it directly.

RULE 6 — MULTI-QUESTION
Answer every question in order. If aux says questions are linked, connect them explicitly.

RULE 7 — WORD BUDGET
W:<n> is a ceiling not a target. Never pad. Exceeding by <20 words is fine if answer is complete."""

# Build the cached system block for Anthropic API
CACHED_SYSTEM_BLOCK = [
    {
        "type": "text",
        "text": ALE_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

# ─────────────────────────────────────────────
# WORD BUDGET FORMULA
# ─────────────────────────────────────────────

def compute_word_budget(result: "DistillResult") -> int:
    """
    Dynamic word budget — lives in user turn, never in system prompt.

    Formula:
        base(urgency) + paths×20 + identifiers×10 − stressor×15 + closing×15

    base:       high=50, medium=90, low=140
    per path:   +20 (each distinct recovery step implied by aux)
    identifier: +10 (each ID/URL that needs exact reproduction)
    stressor:   −15 (user needs speed, not volume)
    closing:    +15 (if aux suggests there are useful unknowns worth offering)
    """
    urgency_base = {"high": 50, "medium": 90, "low": 140, "unknown": 90}
    budget = urgency_base.get(result.urgency, 90)

    # Estimate paths from aux length (rough proxy)
    paths = max(1, min(len(result.aux), 4))
    budget += paths * 20

    # Identifiers in compressed prompt
    import re
    id_patterns = [
        r'\b[A-Z]{2,6}-\d{3,10}\b',
        r'https?://[^\s]+',
        r'\b[A-Z0-9]{8,20}\b',
    ]
    identifier_count = 0
    for pattern in id_patterns:
        identifier_count += len(re.findall(pattern, result.compressed_prompt))
    budget += identifier_count * 10

    # Stressor penalty
    stressor_words = {
        "crying", "screaming", "emergency", "deadline", "urgent",
        "dying", "accident", "immediately", "asap",
    }
    aux_str = " ".join(result.aux).lower()
    if any(w in aux_str for w in stressor_words):
        budget -= 15

    # Closing offer (if aux is rich — implies useful unknowns)
    if len(result.aux) >= 3:
        budget += 15

    return max(40, budget)  # floor at 40 words


# ─────────────────────────────────────────────
# USER TURN BUILDER
# ─────────────────────────────────────────────

def _build_user_turn(result: "DistillResult") -> str:
    """
    Build compact user turn from DistillResult.

    Passthrough format (3 tokens overhead):
        P
        <original prompt>

    Distilled format (single header line + message):
        D|<tone>|<urgency>|W:<budget>|<aux1>;<aux2>;...
        <compressed prompt>

    Fixed overhead of compact header: ~10-12 tokens vs ~40 for verbose format.
    Aux items encoded as semicolon-separated inline — no per-item line breaks.
    """
    if result.passthrough:
        return f"P\n{result.original_prompt}"

    budget     = compute_word_budget(result)
    aux_str    = ";".join(result.aux) if result.aux else ""
    domain_str = f"|{result.domain}" if getattr(result, 'domain', None) else ""
    header     = f"D{domain_str}|{result.tone}|{result.urgency}|W:{budget}|{aux_str}"

    return f"{header}\n{result.compressed_prompt}"


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def build_ale_messages(result: "DistillResult") -> dict:
    """
    Main ALE entry point.
    Returns dict with system= and messages= ready to unpack into
    the Anthropic client.messages.create() call.

    Usage:
        packet = build_ale_messages(distill_result)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            temperature=0.4,
            **packet,
        )
    """
    user_turn = _build_user_turn(result)

    return {
        "system": CACHED_SYSTEM_BLOCK,
        "messages": [
            {"role": "user", "content": user_turn}
        ],
    }


# ─────────────────────────────────────────────
# ROUND TRIP
# ─────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class RoundTripResult:
    """
    Complete record of one full pipeline pass.
    Contains everything needed to evaluate distillation quality:
      - what the user sent
      - what the SLM compressed it to + metadata
      - what the frontier LLM responded
      - token counts and latency at each stage
    """
    # Distillation
    original_prompt:    str
    compressed_prompt:  str
    passthrough:        bool
    passthrough_reason: str
    intent:             str
    tone:               str
    urgency:            str
    aux:                list
    confidence:         float
    distill_latency_ms: float
    distill_tier:       str

    # Frontier LLM
    frontier_response:      str
    input_tokens:           int
    output_tokens:          int
    cache_read_tokens:      int
    cache_write_tokens:     int
    frontier_latency_ms:    float

    # Totals
    total_latency_ms:       float

    def display(self) -> None:
        """Print a clear side-by-side view of the full round trip."""
        sep = "=" * 60

        print(f"\n{sep}")
        print("ROUND TRIP RESULT")
        print(sep)

        # ── Original ──
        print("\n[ ORIGINAL PROMPT ]")
        print(self.original_prompt)

        # ── Distillation ──
        print(f"\n[ DISTILLATION — {self.distill_tier} | {self.distill_latency_ms:.0f}ms ]")
        if self.passthrough:
            print(f"  PASSTHROUGH — {self.passthrough_reason}")
            print("  Original forwarded to frontier LLM unchanged.")
        else:
            print(f"  Compressed : {self.compressed_prompt}")
            print(f"  Intent     : {self.intent}")
            print(f"  Tone       : {self.tone}")
            print(f"  Urgency    : {self.urgency}")
            print(f"  Aux        : {self.aux}")
            print(f"  Confidence : {self.confidence:.2f}")
            orig_words = len(self.original_prompt.split())
            comp_words = len(self.compressed_prompt.split())
            reduction  = (1 - comp_words / orig_words) * 100 if orig_words else 0
            print(f"  Word reduction: {orig_words} → {comp_words} words (~{reduction:.0f}%)")

        # ── Frontier LLM response ──
        print(f"\n[ FRONTIER LLM RESPONSE ]")
        print(self.frontier_response)

        # ── Token economics ──
        print(f"\n[ TOKEN ECONOMICS ]")
        print(f"  Input tokens      : {self.input_tokens}")
        print(f"  Output tokens     : {self.output_tokens}")
        if self.cache_write_tokens:
            print(f"  Cache write       : {self.cache_write_tokens} (1.25x, one-time)")
        if self.cache_read_tokens:
            print(f"  Cache read        : {self.cache_read_tokens} (0.10x, 90% saving)")

        # ── Latency ──
        print(f"\n[ LATENCY ]")
        print(f"  Distillation      : {self.distill_latency_ms:.0f}ms")
        print(f"  Frontier LLM      : {self.frontier_latency_ms:.0f}ms")
        print(f"  Total             : {self.total_latency_ms:.0f}ms")
        print(sep)


def round_trip(
    prompt: str,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 300,
    temperature: float = 0.4,
    spsd_model_path: str = None,
) -> RoundTripResult:
    """
    Full pipeline: distill → ALE packet → frontier LLM → RoundTripResult.

    Args:
        prompt:           raw user prompt
        api_key:          Anthropic API key
        model:            frontier model to use
        max_tokens:       max tokens for frontier response
        temperature:      frontier LLM temperature
        spsd_model_path:  path to Qwen GGUF model (optional override)

    Returns:
        RoundTripResult — everything in one object, call .display() to print.
    """
    import time
    import anthropic
    import spsd_v4

    # ── Step 1: Distill ──────────────────────────────────────
    distill_result = spsd_v4.distill(prompt, model_path=spsd_model_path)

    # ── Step 2: Build ALE packet ─────────────────────────────
    packet = build_ale_messages(distill_result)

    # ── Step 3: Call frontier LLM ────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **packet,
    )
    frontier_latency_ms = (time.time() - t0) * 1000

    frontier_text = response.content[0].text

    # Extract token counts — cache fields may not be present on all responses
    usage = response.usage
    cache_read  = getattr(usage, "cache_read_input_tokens",  0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    total_latency_ms = distill_result.latency_ms + frontier_latency_ms

    return RoundTripResult(
        # Distillation fields
        original_prompt    = distill_result.original_prompt,
        compressed_prompt  = distill_result.compressed_prompt,
        passthrough        = distill_result.passthrough,
        passthrough_reason = distill_result.passthrough_reason or "",
        intent             = distill_result.intent,
        tone               = distill_result.tone,
        urgency            = distill_result.urgency,
        aux                = distill_result.aux,
        confidence         = distill_result.confidence,
        distill_latency_ms = distill_result.latency_ms,
        distill_tier       = distill_result.tier,

        # Frontier LLM fields
        frontier_response   = frontier_text,
        input_tokens        = usage.input_tokens,
        output_tokens       = usage.output_tokens,
        cache_read_tokens   = cache_read,
        cache_write_tokens  = cache_write,
        frontier_latency_ms = frontier_latency_ms,

        total_latency_ms    = total_latency_ms,
    )


# ─────────────────────────────────────────────
# DEBUG PREVIEW
# ─────────────────────────────────────────────

def preview_packet(result: "DistillResult") -> None:
    """Pretty-print the full ALE packet for debugging."""
    user_turn = _build_user_turn(result)
    lines     = user_turn.split('\n')
    tokens    = round(len(user_turn.split()) / 0.75)

    print("\n" + "="*60)
    print("ALE PACKET PREVIEW")
    print("="*60)
    print(f"[USER TURN — ~{tokens} tokens]")
    for line in lines:
        print(f"  {line}")
    if not result.passthrough:
        print(f"\n[WORD BUDGET] {compute_word_budget(result)} words")
    print("="*60)
