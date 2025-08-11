import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import json
import re
import streamlit as st
import pandas as pd
from typing import Any, Dict, Optional
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Hugging Face Analyzer", page_icon="🤔")
st.title("🤔 Kath's AI Personality Analyzer (Free)")
st.caption("Powered by Hugging Face Hosted Inference API — Model: **google/gemma-7b-it** (free)")

# ---------------- API / MODEL ----------------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
MODEL_ID = st.secrets.get("HF_MODEL_ID", os.getenv("HF_MODEL_ID", "google/gemma-7b-it"))

if not HF_TOKEN:
    st.error("🔐 Missing `HF_TOKEN`. Add it in Streamlit → Settings → Secrets.")
    st.stop()

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.5, 0.1)
    max_new_tokens = st.slider("Max output tokens", 256, 1024, 600, 32)
    st.caption("If output truncates, raise tokens. Lower temperature = more precise; higher = more creative.")

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You are a psychologist trained in the Big Five and MBTI.

Return ONLY valid JSON in this schema (no commentary outside JSON):
{
  "emotional_tone": "2-3 sentences summary",
  "big_five": {
    "openness": {"score": 0-100, "reason": "one sentence"},
    "conscientiousness": {"score": 0-100, "reason": "one sentence"},
    "extraversion": {"score": 0-100, "reason": "one sentence"},
    "agreeableness": {"score": 0-100, "reason": "one sentence"},
    "neuroticism": {"score": 0-100, "reason": "one sentence"}
  },
  "mbti": {"type": "e.g., INFP", "reason": "2-3 sentences"},
  "advice": ["short bullet", "short bullet", "short bullet"],
  "recommendations": [
    {"type": "book|career", "title": "item", "reason": "one sentence"},
    {"type": "book|career", "title": "item", "reason": "one sentence"}
  ]
}
Keep scores as integers 0-100. Be concise, accurate, and thoughtful.
""".strip()

def build_prompt(user_text: str) -> str:
    # Single-prompt style for text-generation backends
    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"User:\n{user_text}\n\n"
        "Assistant:\n"
    )

# ---------------- HELPERS ----------------
def _extract_json_block(text: str) -> Optional[str]:
    """Try to pull a JSON object from the model text."""
    m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else None

def parse_response_to_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse model output into dict; returns None if failed."""
    try:
        return json.loads(raw)
    except Exception:
        block = _extract_json_block(raw or "")
        if not block:
            return None
        try:
            return json.loads(block)
        except Exception:
            return None

def safe_int(x, lo=0, hi=100) -> int:
    try:
        v = int(float(x))
        return max(lo, min(hi, v))
    except Exception:
        return 0

def show_big_five(big_five: Dict[str, Any]):
    trait_map = [
        ("openness", "Openness"),
        ("conscientiousness", "Conscientiousness"),
        ("extraversion", "Extraversion"),
        ("agreeableness", "Agreeableness"),
        ("neuroticism", "Neuroticism"),
    ]
    rows = []
    for key, label in trait_map:
        item = big_five.get(key, {}) if isinstance(big_five, dict) else {}
        score = safe_int(item.get("score", 0))
        reason = str(item.get("reason", "") or "")
        rows.append({"Trait": label, "Score": score, "Reason": reason})

    # Pretty render: bars + reasons
    st.subheader("🧭 Big Five Traits")
    for r in rows:
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(f"**{r['Trait']}**")
            st.markdown(f"**{r['Score']}/100**")
        with c2:
            st.progress(r["Score"])
            st.caption(r["Reason"])

def render_results(j: Dict[str, Any]):
    # Emotional tone
    tone = str(j.get("emotional_tone", "") or "")
    if tone:
        st.subheader("🎭 Emotional Tone")
        st.write(tone)

    # Big Five
    bf = j.get("big_five", {})
    if isinstance(bf, dict) and bf:
        show_big_five(bf)

    # MBTI
    mbti = j.get("mbti", {}) if isinstance(j.get("mbti", {}), dict) else {}
    mbti_type = str(mbti.get("type", "") or "")
    mbti_reason = str(mbti.get("reason", "") or "")
    if mbti_type or mbti_reason:
        st.subheader("🔤 MBTI Guess")
        cols = st.columns([1, 5])
        with cols[0]:
            st.markdown(f"### **{mbti_type or '—'}**")
        with cols[1]:
            if mbti_reason:
                st.caption(mbti_reason)

    # Advice
    advice = j.get("advice", [])
    if isinstance(advice, list) and advice:
        st.subheader("💡 Personalized Advice")
        for a in advice:
            st.write("• " + str(a))

    # Recommendations
    recs = j.get("recommendations", [])
    if isinstance(recs, list) and recs:
        st.subheader("📚 Recommendations")
        for r in recs:
            kind = str(r.get("type","") or "").title()
            title = str(r.get("title","") or "")
            reason = str(r.get("reason","") or "")
            st.write(f"• **{kind}: {title}** — {reason}")

    # Download JSON
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps(j, ensure_ascii=False, indent=2),
        file_name="personality_analysis.json",
        mime="application/json",
    )

# ---------------- MAIN UI ----------------
st.markdown("Paste your writing sample below. The analyzer will estimate tone, Big Five traits, a likely MBTI, and give tailored suggestions.")

user_text = st.text_area("✍️ Your text:", height=250, placeholder="A few paragraphs works best…")

analyze = st.button("🔍 Analyze", type="primary")

if analyze:
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()

    with st.spinner("Thinking…"):
        prompt = build_prompt(user_text)
        try:
            raw = client.text_generation(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                # You can optionally add: top_p=0.95, repetition_penalty=1.1
            )
        except HfHubHTTPError as e:
            msg = str(e)
            if "Repository Not Found" in msg or "404" in msg:
                st.error("Model not available on the Hosted Inference API. Try later or switch model.")
            elif "429" in msg:
                st.error("Rate limited. The free API is busy — try again shortly.")
            elif "401" in msg or "Forbidden" in msg:
                st.error("Unauthorized. Check your HF_TOKEN permissions.")
            elif "503" in msg or "504" in msg:
                st.error("Backend unavailable or cold-starting. Try again in a moment.")
            else:
                st.error(f"Hugging Face error: {msg}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # Try JSON parse
    parsed = parse_response_to_json(raw)
    st.subheader("📋 Results")
    if parsed:
        render_results(parsed)
    else:
        st.caption("Couldn’t parse clean JSON — showing raw model output instead.")
        st.markdown(raw)
