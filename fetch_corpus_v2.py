"""
SPSD v4.2 — Corpus Fetch v2 (fetch_corpus_v2.py)
=================================================
Fetches 100 prompts optimised for SPSD hypothesis testing.

Distribution:
  verbose_social       20  synthetic (known-good)
  general_conversational 20  WildChat real users, >25 words
  code_technical       15  CodeFeedback natural language
  multi_intent_linked  15  WildChat/UltraChat 2+ questions
  short_passthrough    10  control group
  single_intent_clear  10  synthetic direct requests
  high_stakes_medical  10  MedQA (all will passthrough in v4.2)

Changes from v1:
  - medical reduced 20→10 (all passthrough, just for passthrough stats)
  - general_conversational increased 16→20, min word count 25
  - multi_intent increased 10→15
  - code_technical reduced 42→15
  - verbose_social increased 18→20

Output: /content/spsd_corpus_v2.csv
"""

import requests, csv, re, sys
from collections import Counter

OUTPUT_PATH = "/content/spsd_corpus_v2.csv"

# ── Quality filters ────────────────────────────────────────────────────────
def is_english(text):
    common = {'the','a','an','is','are','was','i','you','my','to','of',
               'in','for','and','or','can','do','have','not','but','this'}
    return len(common & set(text.lower().split())) >= 2

def is_natural_language(text):
    code_lines = sum(1 for l in text.split('\n')
                     if re.match(r'\s*(def |class |import |function |SELECT |{|}|<)', l))
    return code_lines / max(1, len(text.split('\n'))) < 0.4

def wc(text): return len(text.split())
def clean(text): return re.sub(r'\s+', ' ', text.strip())

seen = set()
def is_dup(text):
    key = text.lower().strip()[:120]
    if key in seen: return True
    seen.add(key); return False

prompts = []

def add(cat, text, source, intent='', domain=''):
    text = clean(text)
    if not text or is_dup(text) or not is_english(text): return False
    prompts.append({'id':'', 'category':cat, 'word_count':wc(text),
                    'prompt':text, 'source':source,
                    'intent_label':intent, 'domain_tag':domain})
    return True

def count(cat): return sum(1 for p in prompts if p['category']==cat)

def fetch(dataset, config='default', split='train', offset=0, length=100):
    try:
        r = requests.get('https://datasets-server.huggingface.co/rows',
                         params={'dataset':dataset,'config':config,
                                 'split':split,'offset':offset,'length':length},
                         timeout=30)
        return r.json().get('rows', [])
    except Exception as e:
        print(f"  [error] {e}"); return []

# ════════════════════════════════════════════════════════════════════════
# SOURCE 1 — VERBOSE SOCIAL (20, synthetic)
# ════════════════════════════════════════════════════════════════════════
print("[1/6] Verbose social — synthetic...")
VERBOSE_SOCIAL = [
    ("Hi I'm so sorry to bother you, I know you must be incredibly busy, but I ordered something three days ago and I still haven't received any shipping confirmation. My kids are really excited about it arriving and I'm starting to worry. The order number is ORD-847291. Could you please help me find out what's happening with it?", "track_order"),
    ("Hello, I hope you're having a good day. I feel really embarrassed reaching out again as I've already contacted support once this week, but I still haven't received my parcel and it's now been six days. I've checked my emails and there's no tracking information at all. Order reference TRK-00492. I'd be so grateful for any update.", "track_order"),
    ("Good morning, I'm really sorry to trouble you but I'm getting quite anxious. I placed an order last Tuesday for my daughter's birthday present and her birthday is this Saturday. I haven't received any dispatch notification and I'm worried it won't arrive in time. Order number ORD-554821. Please could you check on this for me?", "delivery_period"),
    ("I hope this message finds you well. I'm writing because I returned an item about two weeks ago and I still haven't received my refund of £29.99. I have the return tracking confirmation showing it was delivered back to your warehouse on the 4th. I don't want to be a nuisance but I'm starting to get a little worried. Order ORD-221044.", "get_refund"),
    ("Hi there, I really hope you can help me. I am incredibly stressed right now as I was charged twice for the same order last week — I can see two identical charges of £29.99 on my bank statement. I'm on a tight budget and this has caused me real difficulty. The order reference is ORD-847291. I would be so grateful if you could refund the duplicate charge.", "get_refund"),
    ("Hello, please forgive me for reaching out about this again. I submitted a refund request for order ORD-338821 about ten days ago and I received confirmation it would be processed within 5-7 working days. It's now been twelve days and the money still hasn't appeared. I've been a loyal customer for over three years. Could you please chase this up?", "track_refund"),
    ("Hi, I feel really terrible writing this because I know it creates extra work, but I accidentally placed two orders for the same item about five minutes apart. I got an error on the first attempt so I tried again, and it turns out both went through. Orders ORD-771034 and ORD-771041. I only need one — could you cancel the duplicate and refund it?", "cancel_order"),
    ("Good afternoon, I hope I'm not bothering you too much. I placed an order yesterday evening and I've just realised I ordered the wrong size. I ordered a medium but I need a large. The order is ORD-992341 and I don't think it's been dispatched yet. Is there any way to change the size or cancel it so I can reorder? Thank you so much.", "change_order"),
    ("Hello, I'm so embarrassed to admit this but I think I may have fallen for a phishing email earlier today. I received something that looked like it was from your company asking me to verify my account, and I clicked the link before I realised. I've since changed my password but I'm really worried about my account security. Could you check my account sarah@email.com?", "complaint"),
    ("Hi there, I hope you can help me. I've been trying to log into my account for the past two days and I keep getting an error saying my password is incorrect. I've reset it three times now and the same thing keeps happening. I've tried on two different devices and two different browsers. I need to check my order status urgently. My account email is sarah@email.com.", "recover_password"),
    ("Hello, I really hope someone can help me urgently. My order ORD-884521 was marked as delivered yesterday afternoon but it was not left at my address or with any of my neighbours. I've checked with everyone on my street and nobody has it. I'm really worried as the item was quite expensive and it was a gift for my husband's anniversary. Could you investigate this urgently?", "track_order"),
    ("Good morning, I'm so sorry to bother you but I'm in a bit of a panic. I ordered a suit for a job interview that is happening tomorrow morning and the tracking shows it's been sitting at a distribution centre for three days with no update. Order number ORD-667123. Is there anything at all that can be done to get it delivered today? Thank you so much.", "delivery_period"),
    ("Hi, I hope this message is okay to send. I bought a pair of shoes from your website about three weeks ago as a gift for my mother, but unfortunately they don't fit her at all. I kept all the original packaging and the tags are still attached. I'm not sure if I'm still within the return window. Could you please let me know what the return process is?", "complaint"),
    ("Hello, I feel a bit silly asking this but I genuinely cannot find the return instructions anywhere on your website. I need to return an item from order ORD-445521 that arrived damaged — the packaging was clearly crushed in transit and the item inside is broken. I've taken photos. I don't want a refund necessarily, I'd prefer a replacement. Could you guide me through the process?", "complaint"),
    ("Hi there, I hope you don't mind me contacting you about this. I've been a customer for about four years and I've always been really happy with the service. However, I noticed this morning that I've been charged for a Premium subscription even though I downgraded to the Standard plan last month. I have the confirmation email. The charge is £29.99 dated yesterday. Could you refund the difference?", "get_refund"),
    ("Good evening, I really hope someone can help me. I've been trying to cancel my subscription for the past two weeks and every time I go to the cancellation page it just shows an error. I've tried on my phone, my laptop, and my tablet. I don't want to be charged again next month. My account email is sarah@email.com. Is there a way to cancel directly through you?", "cancel_order"),
    ("Hello, I want to start by saying I've always had excellent service from your company until now, which is why I'm so disappointed. I've contacted your support team three times over the past two weeks about a missing item from my order ORD-338821 and each time I've been told someone will follow up within 48 hours. No one has. I'm not trying to be difficult but I'm really frustrated.", "complaint"),
    ("Hi, I'm writing because I'm honestly at my wits' end. My elderly mother relies on a repeat delivery from your service and the last two months in a row it has either arrived late or with items missing. Last month it was her medication substitute and this month three items are missing from order ORD-992104. She's 84 and doesn't have anyone else to help her. Could someone senior look at her account?", "complaint"),
    ("Good morning, I really hope you can help. I placed an order six days ago for a wedding anniversary gift and it still hasn't arrived. My anniversary is tomorrow and I'm getting quite distressed. I've been tracking it and it hasn't moved in four days. Order number ORD-773210. Please could you look into this as urgently as possible? I would be so grateful.", "delivery_period"),
    ("Hi, I know you must receive so many messages like this but I really need help. My order ORD-558821 arrived today but the wrong item was in the box — I ordered a blue version but received a red one. I need the correct item by Friday for an important event. Could you please arrange an urgent replacement? I'm happy to return the incorrect one immediately.", "problem_with_order"),

    ("Hello, I'm really sorry to be writing to you about this again. I placed an order for my son's graduation gift three weeks ago and it still hasn't arrived. His graduation ceremony is next week and I'm absolutely panicking. I've tried tracking it online but the page just shows an error. Order reference ORD-662210. Could you please look into this as a matter of urgency? I'd be incredibly grateful.", "delivery_period"),
    ("Hi there, I hope you don't mind me getting in touch directly. I've been trying to get a resolution on a damaged item for the past month now. I've spoken to three different agents and each time I'm told a replacement will be sent within 7 days. It never arrives. I'm really at the end of my patience now. The original order was ORD-774432. Could someone senior please take ownership of this?", "complaint"),
    ("Good morning, I feel terrible about bothering you on a weekend but I'm quite desperate. My mother is 79 and lives alone and she relies on her weekly delivery arriving on time. This week it's two days late and she has run out of several essential items including her meal supplements. Could you please check what's happened to order ORD-881122 and whether it can still be delivered today or tomorrow?", "delivery_period"),
    ("Hi, I know this is probably a very common issue but I'm hoping you can help me. I bought a laptop bag as a Christmas gift for my partner and I've just discovered it has a manufacturing fault — one of the zips is completely broken and the bag has never been used. I have the receipt and it's within the warranty period. Order number ORD-445678. How do I arrange a return or replacement please?", "complaint"),
    ("Hello, I'm writing because I'm genuinely confused and a little worried. I received an email yesterday saying my order ORD-993421 has been delivered and signed for, but I was at home all day and nobody knocked. I've checked with my neighbours and the local post office and nobody has it. The item was worth over £80 and was a birthday gift. Please could you investigate this urgently?", "track_order"),
    ("Hi there, I really hope someone can help me resolve this. I've been trying to update my billing details on my account for the past week because my old card has expired. Every time I try to save the new card details I get an error message saying 'payment method not accepted'. I've tried three different cards and the same thing happens. My account email is sarah@email.com. This is stopping me from placing a new order.", "complaint"),
    ("Good afternoon, I'm so sorry to trouble you but I need to raise something that happened with my last order. I ordered three items but only two arrived in the package. The missing item is the most expensive one — it was a handbag worth £65. I've checked the invoice and all three items are listed as included. Order ORD-557234. Could you please arrange to send the missing item or issue a partial refund?", "problem_with_order"),
]

for text, intent in VERBOSE_SOCIAL:
    add("verbose_social", text, "synthetic_spsd_pattern", intent, "SUPPORT")

print(f"  verbose_social: {count('verbose_social')}/20")

# ════════════════════════════════════════════════════════════════════════
# SOURCE 2 — SINGLE INTENT CLEAR (10, synthetic)
# ════════════════════════════════════════════════════════════════════════
print("[2/6] Single intent clear — synthetic...")
SINGLE_INTENT = [
    ("I need to cancel order ORD-847291 placed this morning before it ships please.", "cancel_order"),
    ("Can you check the delivery status for order TRK-00492 which was supposed to arrive two days ago?", "track_order"),
    ("I'd like a refund for order ORD-338821 — the item arrived damaged and I've sent photos to support.", "get_refund"),
    ("Please change the delivery address for order ORD-554821 to 14 Birchwood Lane, Manchester M14 6PQ.", "change_shipping_address"),
    ("I can't log into my account — I've reset my password twice and still getting invalid credentials on sarah@email.com.", "recover_password"),
    ("I was charged £29.99 twice for the same order on the 12th — please refund the duplicate charge.", "get_refund"),
    ("Please cancel my Premium Monthly subscription at the end of this billing cycle.", "cancel_order"),
    ("The item in order ORD-771034 is the wrong size — I ordered large but received medium, how do I exchange it?", "change_order"),
    ("My account shows a pending charge of £29.99 but I didn't place any order recently — please check if this is unauthorised.", "complaint"),
    ("I need to update the email address on my account from sarah@email.com to sarah.jones@email.com.", "edit_account"),

    ("I need to return order ORD-445521 — wrong item delivered, I ordered blue but received red, collection please.", "problem_with_order"),
    ("Can you confirm whether order ORD-992341 qualifies for free returns or if I need to pay postage?", "check_refund_policy"),
    ("Please update my delivery instructions for all future orders to leave with neighbour at number 12.", "change_shipping_address"),
    ("I'd like to downgrade my account from Premium to Standard plan effective from next billing date.", "cancel_order"),
    ("Can you resend the order confirmation email for ORD-847291 to sarah@email.com as I accidentally deleted it?", "get_invoice"),
]
for text, intent in SINGLE_INTENT:
    add("single_intent_clear", text, "synthetic_spsd_pattern", intent, "SUPPORT")

print(f"  single_intent_clear: {count('single_intent_clear')}/10")

# ════════════════════════════════════════════════════════════════════════
# SOURCE 3 — WILDCHAT: general_conversational (20, >25 words)
#            + multi_intent_linked (15, 2+ questions)
# ════════════════════════════════════════════════════════════════════════
print("[3/6] WildChat — general conversational + multi intent...")
general_target = 30
multi_target   = 25

for offset in range(0, 5000, 100):
    if count('general_conversational') >= general_target and count('multi_intent_linked') >= multi_target:
        break
    rows = fetch("allenai/WildChat-4.8M", length=100, offset=offset)
    for row in rows:
        r = row["row"]
        if r.get("language","English") != "English": continue
        if r.get("toxic", False): continue
        conv = r.get("conversation", [])
        first = next((t for t in conv if t.get("role")=="user"), None)
        if not first: continue
        text = first.get("content","").strip()
        if not is_english(text) or not is_natural_language(text): continue
        if text.startswith("```"): continue
        words = wc(text)

        is_multi = (text.count("?") >= 2 or
                    bool(re.search(r'\b(two things|two questions|also|secondly|'
                                   r'and also|another question|follow.?up)\b',
                                   text, re.IGNORECASE)))

        # multi_intent: 25-150 words, 2+ questions
        if is_multi and 25 <= words <= 150 and count('multi_intent_linked') < multi_target:
            add("multi_intent_linked", text, "allenai/WildChat-4.8M")

        # general_conversational: 25-100 words (min 25 to ensure distillable)
        elif not is_multi and 25 <= words <= 100 and count('general_conversational') < general_target:
            add("general_conversational", text, "allenai/WildChat-4.8M")

    print(f"  offset={offset:4d} | general={count('general_conversational'):2d}/{general_target} "
          f"multi={count('multi_intent_linked'):2d}/{multi_target}", end='\r')

print(f"\n  general_conversational: {count('general_conversational')}/{general_target}")
print(f"  multi_intent_linked   : {count('multi_intent_linked')}/{multi_target}")

# UltraChat top-up for multi if needed
if count('multi_intent_linked') < multi_target:
    needed = multi_target - count('multi_intent_linked')
    print(f"  UltraChat top-up: need {needed} more multi...")
    for offset in range(0, 1000, 50):
        rows = fetch("HuggingFaceH4/ultrachat_200k",
                     split="train_sft", length=50, offset=offset)
        for row in rows:
            msgs  = row["row"].get("messages", [])
            first = next((m for m in msgs if m.get("role")=="user"), None)
            if not first: continue
            text = first.get("content","").strip()
            if wc(text) < 25 or wc(text) > 150: continue
            if text.count("?") < 2: continue
            if not is_natural_language(text): continue
            add("multi_intent_linked", text, "HuggingFaceH4/ultrachat_200k")
        if count('multi_intent_linked') >= multi_target: break

# ════════════════════════════════════════════════════════════════════════
# SOURCE 4 — CODEFEEDBACK: code_technical (15, natural language only)
# ════════════════════════════════════════════════════════════════════════
print("[4/6] CodeFeedback — code technical...")
for offset in range(0, 1200, 50):
    if count('code_technical') >= 25: break
    rows = fetch("m-a-p/CodeFeedback-Filtered-Instruction", length=50, offset=offset)
    for row in rows:
        text = row["row"].get("query","")
        if wc(text) < 15 or wc(text) > 150: continue
        if not is_natural_language(text): continue
        if text.strip().startswith("```"): continue
        add("code_technical", text, "m-a-p/CodeFeedback-Filtered-Instruction",
            "technical_help", "CODE")
    print(f"  code={count('code_technical')}/15", end='\r')
print(f"\n  code_technical: {count('code_technical')}/15")

# ════════════════════════════════════════════════════════════════════════
# SOURCE 5 — MEDQA: high_stakes_medical (10, all will passthrough)
# ════════════════════════════════════════════════════════════════════════
print("[5/6] MedQA — medical (passthrough control)...")
for offset in range(0, 200, 10):
    if count('high_stakes_medical') >= 15: break
    rows = fetch("GBaker/MedQA-USMLE-4-options", length=10, offset=offset)
    for row in rows:
        text = row["row"].get("question","")
        if wc(text) < 20: continue
        add("high_stakes_medical", text,
            "GBaker/MedQA-USMLE-4-options", "medical_question", "MEDICAL")
print(f"  high_stakes_medical: {count('high_stakes_medical')}/10")

# ════════════════════════════════════════════════════════════════════════
# SOURCE 6 — SHORT PASSTHROUGH (10, control group)
# ════════════════════════════════════════════════════════════════════════
print("[6/6] Short passthrough — control group...")
BITEXT_PLACEHOLDER_FILLS = {
    r'\{\{Order Number\}\}':   'ORD-847291',
    r'\{\{Account Number\}\}': 'ACC-10047',
    r'\{\{Tracking Number\}\}':'TRK-00492',
    r'\{\{Product\}\}':        'the item I ordered',
    r'\{\{Name\}\}':           'Sarah',
    r'\{\{Amount\}\}':         '£29.99',
}
def fill(text):
    for p, v in BITEXT_PLACEHOLDER_FILLS.items():
        text = re.sub(p, v, text, flags=re.IGNORECASE)
    return None if re.search(r'\{\{.*?\}\}', text) else text

for offset in range(0, 2000, 100):
    if count('short_passthrough') >= 15: break
    rows = fetch("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                 length=100, offset=offset)
    for row in rows:
        text = fill(row["row"].get("instruction",""))
        if not text: continue
        if wc(text) <= 12:
            add("short_passthrough", text,
                "bitext/Bitext-customer-support",
                row["row"].get("intent",""), "ORDER")
print(f"  short_passthrough: {count('short_passthrough')}/10")

# ════════════════════════════════════════════════════════════════════════
# TRIM TO TARGETS AND ASSIGN IDS
# ════════════════════════════════════════════════════════════════════════
cat_targets = {
    "verbose_social":        25,
    "single_intent_clear":   15,
    "general_conversational":30,
    "multi_intent_linked":   25,
    "code_technical":        25,
    "high_stakes_medical":   15,
    "short_passthrough":     15,
}
cat_order = list(cat_targets.keys())

final = []
for cat in cat_order:
    subset = [p for p in prompts if p["category"]==cat]
    final.extend(subset[:cat_targets[cat]])

# Top up to 100 if any category fell short
remaining = [p for p in prompts if p not in final]
while len(final) < 150 and remaining:
    final.append(remaining.pop(0))

for i, p in enumerate(final, 1):
    p["id"] = f"P{i:03d}"

# ════════════════════════════════════════════════════════════════════════
# WRITE
# ════════════════════════════════════════════════════════════════════════
fields = ["id","category","word_count","prompt","source","intent_label","domain_tag"]
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(final)

# ════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"CORPUS v2 (150 prompts) — {len(final)} fetched → {OUTPUT_PATH}")
print(f"{'='*60}")
cats_final = Counter(p["category"] for p in final)
for cat in cat_order:
    n   = cats_final.get(cat, 0)
    tgt = cat_targets[cat]
    wcs = [p["word_count"] for p in final if p["category"]==cat]
    ok  = "✓" if n >= tgt*0.9 else "⚠"
    print(f"  {ok} {cat:28s} {n:3d}/{tgt}  avg={sum(wcs)//max(len(wcs),1)}w")

support_cats = {"verbose_social","single_intent_clear","short_passthrough"}
n_support = sum(cats_final.get(c,0) for c in support_cats)
n_other   = len(final) - n_support
print(f"\n  Support  : {n_support} ({n_support/len(final)*100:.0f}%)")
print(f"  Other    : {n_other} ({n_other/len(final)*100:.0f}%)")
print(f"\nNext → %run run_spsd_only_v2.py")
