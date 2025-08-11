import os
import streamlit as st
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 Kath's AI Personality Analyzer")
st.subheader("Hugging Face Model gpt-oss-20b")

st.write(
    "Runs a psychology-style analysis (Big Five + MBTI + advice) using a Hugging Face Inference Endpoint "
    "or the Hosted Inference API if available."
)

# ----------------- CONFIG / SECRETS -----------------

HF_TOKEN = os.getenv("HF_TOKEN")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")  # e.g. https://abc123.us-east-1.aws.endpoints.huggingface.cloud
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b")

if HF_ENDPOINT_URL:
    # ✅ Correct for Inference Endpoint
    client = InferenceClient(model=HF_ENDPOINT_URL, token=HF_TOKEN)
    # or: InferenceClient(base_url=HF_ENDPOINT_URL, api_key=HF_TOKEN)
else:
    # Hosted (serverless) Inference API — may not be enabled for this model
    client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)
    st.caption(f"Using Hosted Inference API for `{HF_MODEL_ID}` (may not be enabled for this model).")

# ----------------- STREAMLIT BODY UI -----------------

st.divider()

st.markdown("Paste your writing sample below and get a quick personality readout.")
user_text = st.text_area("✍️ Your text:", height=240)

temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7, 0.1)
max_new_tokens = st.slider("Max output tokens", 128, 1024, 600, 32)
st.caption("Tip: If output truncates, raise ‘Max output tokens’. Higher temperature = more creative, lower = more factual.")

system_prompt = (
    "You are a psychologist trained in the Big Five and MBTI. Analyze the user's writing and return:\n"
    "1) Emotional tone\n"
    "2) Big Five personality trait estimates (O,C,E,A,N) with 0–100 scores + 1-line rationale each\n"
    "3) A likely MBTI type with a 2–3 sentence justification\n"
    "4) Personalized advice plus 2 book OR career recommendations with one-sentence reasons\n"
    "Be concise, accurate, and thoughtful. Use clear section headings."
)

def build_prompt(user_text: str) -> str:
    # Simple “chat in a single prompt” pattern for text-generation endpoints
    return (
        f"System:\n{system_prompt}\n\n"
        f"User:\n{user_text}\n\n"
        "Assistant:\n"
    )

def generate_text(prompt: str):
    """
    Stream tokens from HF Inference. Works with Inference Endpoints and Hosted Inference API
    that support text-generation (TGI or compatible backends).
    """
    try:
        # stream=True yields an iterator of string chunks
        for chunk in client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
            # safety knobs if supported by backend:
            # do_sample=True, top_p=0.95, repetition_penalty=1.1
        ):
            yield chunk
    except HfHubHTTPError as e:
        # Surface common cases clearly
        msg = str(e)
        if "Repository Not Found" in msg or "404" in msg:
            yield "\n\n**Error:** Model/endpoint not found. If you’re not using an endpoint, "
            yield "ensure the model ID is correct and publicly accessible, or create an Inference Endpoint.\n"
        elif "401" in msg or "Forbidden" in msg:
            yield "\n\n**Error:** Unauthorized. Check HF_TOKEN scope and that your token has access to the endpoint/model.\n"
        elif "429" in msg:
            yield "\n\n**Error:** Rate limited. Try again later or increase your plan’s limits.\n"
        elif "503" in msg:
            yield "\n\n**Error:** Backend unavailable. The model may be spinning up; retry in a moment.\n"
        else:
            yield f"\n\n**HF Error:** {msg}\n"
    except Exception as e:
        yield f"\n\n**Unexpected error:** {e}\n"

if st.button("Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Thinking…"):
            prompt = build_prompt(user_text)
            placeholder = st.empty()
            output_accum = ""

            for token in generate_text(prompt):
                output_accum += token
                placeholder.markdown(output_accum)

        st.success("Done!")
