import re
from typing import Any, Dict, List, Tuple, Optional

# Business-level rule engine that only uses fields from prediction JSON records.
# It does NOT change model labels; it adds "kbs" fields for strategy actions.

_WORD_BOUNDARY = r"(?<!\w){}(?!\w)"

def _wb(word: str) -> str:
    """Regex pattern for matching a standalone word (avoid substring hits like 'ulat' in 'circulation')."""
    return _WORD_BOUNDARY.format(re.escape(word))


def _phrase(phrase: str) -> str:
    """Regex pattern for matching a multi-word phrase using word boundaries per token.

    Tokens are separated by one or more non-word chars (spaces/punct). Whole-word matching avoids
    substring hits (e.g. 'ig' in 'might').
    """
    tokens = [tok for tok in re.split(r"\s+", phrase.strip()) if tok]
    if not tokens:
        return r"$"
    escaped = [re.escape(tok) for tok in tokens]
    inner = r"\W+".join(escaped)
    return _WORD_BOUNDARY.format(inner)

def _has_any(text: str, patterns: List[str]) -> Tuple[bool, Optional[str]]:
    """Return (hit, matched_string). matched_string is the literal substring matched in text."""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return True, m.group(0)
    return False, None

def _norm_text(rec: Dict[str, Any]) -> str:
    t = (rec.get("text_normalized") or rec.get("text") or rec.get("review") or "")
    return str(t)

def _get_aspects(rec: Dict[str, Any]) -> List[str]:
    a = rec.get("model_pred_aspects")
    if isinstance(a, list):
        return [str(x) for x in a if x]
    a = rec.get("nli_aspects")
    if isinstance(a, list):
        return [str(x) for x in a if x]
    return []

def _get_primary_type(rec: Dict[str, Any]) -> str:
    t = rec.get("model_pred_comment_type") or rec.get("nli_comment_type") or rec.get("comment_type") or ""
    return str(t) if t else ""

def _get_overall_conf(rec: Dict[str, Any]) -> Optional[float]:
    mc = rec.get("model_confidence") or {}
    v = mc.get("overall")
    if v is None:
        return None
    try:
        if isinstance(v, str) and v.strip().lower().startswith("overall="):
            v = v.split("=", 1)[1].strip()
        return float(v)
    except Exception:
        return None

def apply_business_rules(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a shallow-copied record with added key:
      rec["kbs"] = {
          "priority": "P0"|"P1"|"P2"|"P3",
          "owner_team": str,
          "escalation_level": "none"|"supervisor"|"management",
          "business_tags": [str,...],
          "recommended_action": str,
          "rule_trace": [{"rule_id":..., "matched":..., "explain":...}, ...]
      }
    """
    out = dict(rec)
    text = _norm_text(rec)
    text_l = text.lower()
    aspects = set(_get_aspects(rec))
    ctype = _get_primary_type(rec).lower()
    conf = _get_overall_conf(rec)

    tags: List[str] = []
    trace: List[Dict[str, Any]] = []
    priority = "P3"
    owner = "CX"
    escalation = "none"
    action = "Review in dashboard and triage if needed."

    def fire(rule_id: str, matched: str, explain: str):
        trace.append({"rule_id": rule_id, "matched": matched, "explain": explain})

    # --- Rule groups (keyword-only; prediction JSON only) ---

    # P0: food safety / health risk
    food_safety_patterns = [
        # Illness / symptoms (ID + EN) - prefer whole-word patterns to avoid substring matches.
        r"\bkeracun\w*\b",
        _wb("muntah"),
        _wb("mual"),
        _wb("diare"),
        _wb("mencret"),
        _phrase("sakit perut"),
        _phrase("stomach ache"),
        r"\bdiarrhea\b",
        r"\bvomit\b",
        r"\bthrow\w*\b\W+up\b",
        r"\bfood\W+poison\w*\b",
        _phrase("made me sick"),
        _phrase("got sick"),

        # Contamination / foreign object (ID + EN)
        _phrase("bau basi"),
        _wb("basi"),
        _phrase("ada kaca"),
        _wb("kaca"),
        r"\bglass\b",
        _wb("jamur"),
        r"\bmold\w*\b",
        _wb("belatung"),
        r"\bmaggot\w*\b",
        _wb("ulat"),
        r"\bworm\w*\b",
        _phrase("ada rambut"),
        _wb("rambut"),
        r"\bhair\b",

        # Undercooked in a complaint context (avoid 'raw vegetables is good' false positives)
        _phrase("masih mentah"),
        r"\bundercook\w*\b",
        r"\bnot\W+cook\w*\b",
        r"\bstill\W+raw\b\W+(chicken|meat|fish|egg)\b",
        r"\braw\b\W+(chicken|meat|fish|egg)\b",
    ]
    hit, pat = _has_any(text, food_safety_patterns)
    if hit and ("food" in aspects or "price" in aspects or True):
        priority = "P0"
        owner = "Kitchen/QC"
        escalation = "management"
        tags.extend(["food_safety_risk"])
        action = "Immediate QC check: isolate suspected batch, verify storage/temperature, and follow up with customer."
        fire("R_ESCALATE_FOOD_SAFETY", pat or "keyword", "Health/safety signal detected in text.")
        # do not return; allow adding secondary tags below

    # P0/P1: reputation/legal escalation
    rep_patterns = [
        r"\bviral\b",
        _wb("tiktok"),
        _wb("twitter"),
        r"\bx\b",
        _wb("instagram"),
        r"\big\b",
        _wb("boikot"),
        _wb("lapor"),
        _wb("polisi"),
        _wb("bpom"),
        _wb("dinas"),
    ]
    hit, pat = _has_any(text, rep_patterns)
    if hit:
        if priority != "P0":
            priority = "P1"
        owner = "CX/Comms"
        escalation = "management"
        tags.extend(["reputation_risk"])
        action = "Escalate to CX/Comms: prepare response, gather facts, and coordinate with ops."
        fire("R_ESCALATE_REPUTATION_EVENT", pat or "keyword", "Reputation/legal escalation signal detected.")

    # P1: queue / delay
    queue_patterns = [
        _phrase("antri panjang"),
        _phrase("antre panjang"),
        _phrase("nunggu lama"),
        _phrase("menunggu lama"),
        _phrase("lama banget"),
        _wb("delay"),
        _wb("telat"),
        _phrase("served late"),
        _phrase("too slow"),
        _phrase("wait too long"),
        _phrase("long wait"),
        _wb("queue"),
        _wb("line"),
    ]
    hit, pat = _has_any(text, queue_patterns)
    if hit and ("service" in aspects or ctype == "complaint" or True):
        if priority not in ("P0",):
            priority = "P1"
        owner = "Ops/Store"
        escalation = escalation if escalation != "none" else "supervisor"
        tags.extend(["queue_issue"])
        if action.startswith("Review in dashboard"):
            action = "Ops check: review staffing and service flow (cashier/kitchen handoff) for peak hours."
        fire("R_QUEUE_AND_WAIT_TIME", pat or "keyword", "Queue/wait-time signal detected.")

    # P1: staff attitude
    attitude_patterns = [
        _wb("jutek"), _wb("kasar"), _wb("cuek"), _wb("ketus"),
        _phrase("tidak ramah"), _phrase("ga ramah"), _phrase("nggak ramah"),
        _wb("rude"),
        _wb("unfriendly"),
        _wb("impolite"),
        _phrase("not polite"),
    ]
    hit, pat = _has_any(text, attitude_patterns)
    if hit and ("service" in aspects or ctype == "complaint" or True):
        if priority not in ("P0",):
            priority = "P1"
        owner = "Ops/Store"
        escalation = escalation if escalation != "none" else "supervisor"
        tags.extend(["staff_attitude"])
        if action.startswith("Review in dashboard"):
            action = "Ops coaching: refresh hospitality SOP and monitor repeated incidents."
        fire("R_STAFF_ATTITUDE", pat or "keyword", "Staff attitude signal detected.")

    # P1: hygiene / cleanliness
    hygiene_patterns = [
        _phrase("kebersihan kurang"),
        _phrase("kurang bersih"),
        _phrase("toilet kotor"),
        _phrase("meja kotor"),
        _wb("jorok"),
        _wb("kotor"),
        _wb("bau"),
        _wb("lengket"),
        _wb("tikus"),
        _wb("kecoa"),
        r"\bcockroach\w*\b",
        r"\brat\w*\b",
        _wb("dirty"),
        _wb("filthy"),
        r"\bsmell\w*\b",
        r"\bstink\w*\b",
        _wb("sticky"),
        _wb("hygiene"),
        _wb("cleanliness"),
        _wb("restroom"),
        _wb("toilet"),
    ]
    hit, pat = _has_any(text, hygiene_patterns)
    if hit and ("place_ambience" in aspects or ctype == "complaint" or True):
        if priority not in ("P0",):
            priority = "P1"
        owner = "Ops/Store"
        escalation = escalation if escalation != "none" else "supervisor"
        tags.extend(["hygiene_issue"])
        if action.startswith("Review in dashboard"):
            action = "Ops checklist: enforce cleaning checklist and audit store hygiene."
        fire("R_HYGIENE_PLACE", pat or "keyword", "Hygiene/cleanliness signal detected.")

    # P1: order accuracy (missing/wrong items)
    order_patterns = [
        _phrase("salah pesanan"),
        _phrase("pesanan salah"),
        _phrase("tidak sesuai"),
        _phrase("ga sesuai"),
        _phrase("nggak sesuai"),
        _phrase("belum dikasih"),
        _phrase("tidak dikasih"),
        r"\bmissing\b",
        _phrase("wrong order"),
        _wb("forgot"),
        _phrase("not included"),
        _phrase("left out"),
        # 'kurang' must be tied to an order/item context to avoid hitting hygiene phrases
        r"\bkurang\b\W+(satu|1|dua|2|tiga|3|item|porsi|topping|sambal|saus|sendok|garpu|nasi|minum|drink)\b",
        r"\b(order|pesanan)\b[\s\S]{0,40}\b(incomplete|kurang)\b",
    ]
    hit, pat = _has_any(text, order_patterns)
    if hit and (ctype == "complaint" or True):
        if priority not in ("P0",):
            priority = "P1"
        owner = "Ops/Store"
        escalation = escalation if escalation != "none" else "supervisor"
        tags.extend(["order_accuracy"])
        if action.startswith("Review in dashboard"):
            action = "Ops check: review packing/checklist and confirm order handoff procedure."
        fire("R_ORDER_ACCURACY", pat or "keyword", "Order accuracy issue detected.")

    # P2: food quality (taste/temperature/texture) - not safety
    quality_patterns = [
        r"asin", r"tawar", r"pahit", r"kurang rasa", r"lack of taste", r"dingin", r"cold",
        r"keras", r"over spicy", r"kebanyakan gula", r"gosong", r"burnt"
    ]
    hit, pat = _has_any(text, quality_patterns)
    if hit and ("food" in aspects or True):
        if priority not in ("P0", "P1"):
            priority = "P2"
        owner = "Kitchen/QC"
        tags.extend(["food_quality"])
        if action.startswith("Review in dashboard"):
            action = "Kitchen/QC: review recipe consistency, serving temperature, and prep SOP."
        fire("R_FOOD_QUALITY", pat or "keyword", "Food quality signal detected (non-safety).")

    # P2: pricing/value complaint
    price_patterns = [r"mahal", r"kemahalan", r"overprice", r"expensive", r"ga worth", r"nggak worth", r"worth it", r"harga naik", r"pajak", r"service charge"]
    hit, pat = _has_any(text, price_patterns)
    if hit and ("price" in aspects or ctype == "complaint" or True):
        if priority not in ("P0", "P1"):
            priority = "P2"
        owner = "Pricing/Finance"
        tags.extend(["overpricing_claim"])
        if action.startswith("Review in dashboard"):
            action = "Pricing review: check price parity, promotions, and value communication."
        fire("R_PRICING_VALUE_COMPLAINT", pat or "keyword", "Pricing/value signal detected.")

    # P3: actionable suggestion
    suggestion_patterns = [r"sebaiknya", r"tolong", r"mohon", r"sarannya", r"lebih baik", r"harusnya", r"coba\b", r"mending", r"please"]
    hit, pat = _has_any(text, suggestion_patterns)
    if hit and (ctype == "suggestion" or True):
        if priority not in ("P0", "P1"):
            priority = "P3"
        # owner by aspect if possible
        if "service" in aspects:
            owner = "Ops/Store"
        elif "food" in aspects:
            owner = "Kitchen/QC"
        elif "place_ambience" in aspects:
            owner = "Ops/Store"
        elif "price" in aspects:
            owner = "Pricing/Finance"
        tags.extend(["improvement_request"])
        if action.startswith("Review in dashboard"):
            action = "Log to improvement backlog and review feasibility."
        fire("R_SUGGESTION_ACTIONABLE", pat or "keyword", "Suggestion cue detected.")

    # Promoter signal (growth)
    promoter_patterns = [r"rekomen", r"recommended", r"langganan", r"best", r"mantap", r"enak banget", r"love", r"will be back", r"balik lagi"]
    hit, pat = _has_any(text, promoter_patterns)
    if hit and (ctype == "praise" or True):
        tags.extend(["promoter"])
        if priority == "P3" and owner == "CX":
            owner = "CX/Comms"
        if action.startswith("Review in dashboard"):
            action = "Consider UGC/testimonial capture and loyalty nudges."
        fire("R_PROMOTER_SIGNAL", pat or "keyword", "Promoter/advocacy cue detected.")

    # Confidence-aware nudge: low confidence or disagreement
    agreement = (rec.get("agreement") or {}).get("overall") or ""
    if (conf is not None and conf < 0.35) or (agreement == "none"):
        tags.append("needs_review")
        fire("R_LOW_CONF_OR_DISAGREE", "overall_conf/agree", "Low overall confidence or NLI-model disagreement; recommend manual review.")
        if priority not in ("P0",):
            # do not automatically escalate; keep as triage
            if priority == "P3":
                priority = "P2"

    # de-duplicate tags
    tags_unique = []
    for t in tags:
        if t and t not in tags_unique:
            tags_unique.append(t)

    out["kbs"] = {
        "priority": priority,
        "owner_team": owner,
        "escalation_level": escalation,
        "business_tags": tags_unique,
        "recommended_action": action,
        "rule_trace": trace,
    }
    return out
