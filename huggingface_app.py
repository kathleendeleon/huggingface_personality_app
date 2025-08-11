import os
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

st.set_page_config(page_title="Hugging Face Analyzer", page_icon="🤔")
st.title("🤔 Kath's AI Personality Analyzer")

st.write(
    "This version runs on Hugging Face’s free Hosted Inference API using "
    "**mistralai/Mistral-7B-Instruct-v0.2**. It’s slower than OpenAI but free."
)

# --------- Config / Secrets ----------
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))  # Free HF access token (read-only)
MODEL_ID = st.secrets.get("HF_MODEL_ID", os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2"))

if not HF_TOKEN:
    st.error("🔐 Missing HF_TOKEN. Add it in Streamlit → Settings → Secrets.")
    st.stop()

# Hosted Inference API (serverless). No endpoint needed.
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
st.caption(f"Using Hosted Inference API: `{MODEL_ID}`")

st.divider()
st.markdown("Paste your writing sample below to get a quick personality readout (Big Five + MBTI + advice).")

# UI controls
text = st.text_area("✍️ Your text:", height=240, placeholder="Paste a paragraph or two…")
c1, c2 = st.columns(2)
with c1:
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.6, 0.1)
with c2:
    max_new_tokens = st.slider("Max output tokens", 256, 1024, 600, 32)
st.caption("Tip: If output truncates, raise ‘Max output tokens’. Lower temperature = more precise, higher = more creative.")

# Prompt template (single-turn “chat” in one prompt, suited for text-generation backends)
SYSTEM_PROMPT = (
    "You are a psychologist trained in the Big Five and MBTI. Analyze the user's writing and return:\n"
    "1) Emotional tone (2–3 sentences)\n"
    "2) Big Five trait estimates (O, C, E, A, N) with 0–100 scores and a one-line rationale each\n"
    "3) A likely MBTI type with a brief 2–3 sentence justification\n"
    "4) Personalized suggestions and two book OR career recommendations with one-sentence reasons\n"
    "Be concise, accurate, and thoughtful. Use clear section headings and bullet points when helpful."
)

def build_prompt(user_text: str) -> str:
    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"User:\n{user_text}\n\n"
        "Assistant:\n"
    )

def analyze_text(prompt: str) -> str:
    """Call HF Hosted Inference API. Use non-streaming for broad compatibility."""
    try:
        return client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            # You can add: top_p=0.95, repetition_penalty=1.1 if desired
        )
    except HfHubHTTPError as e:
        msg = str(e)
        if "Repository Not Found" in msg or "404" in msg:
            return ("**Error:** Model not available on Hosted Inference API. "
                    "This usually means the free serverless endpoint isn't wired for this model.\n"
                    "Try later or switch to a different free model (e.g., `tiiuae/falcon-7b-instruct`).")
        if "429" in msg:
            return "**Error:** Rate limited. The free API is busy — please try again shortly."
        if "401" in msg or "Forbidden" in msg:
            return "**Error:** Unauthorized. Check your HF_TOKEN’s permissions."
        if "503" in msg or "504" in msg:
            return "**Error:** Backend unavailable or cold-starting. Try again in a moment."
        return f"**HF Error:** {msg}"
    except Exception as e:
        return f"**Unexpected error:** {e}"

if st.button("🔍 Analyze"):
    if not text.strip():
        st.warning("Please paste some text to analyze.")
    else:
        with st.spinner("Thinking…"):
            prompt = build_prompt(text)
            output = analyze_text(prompt)
        st.subheader("📋 Results")
        st.markdown(output)

