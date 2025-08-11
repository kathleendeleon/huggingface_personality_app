import streamlit as st
import os
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Hugging Face Analyzer", page_icon="🤔")
st.title("🤔 Kath's AI Personality Analyzer (Free)")

st.write("Powered by Hugging Face — Model: [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it) (free)")

# --- API KEY ---
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Streamlit Cloud Secrets
if not HF_TOKEN:
    st.error("🔐 Missing Hugging Face API token. Please set HF_TOKEN in Streamlit Cloud secrets.")
    st.stop()

client = InferenceClient(model="google/gemma-7b-it", token=HF_TOKEN)

# --- UI ---
st.markdown("Paste your writing sample below and discover your personality profile based on tone, traits, and helpful suggestions.")
user_text = st.text_area("✍️ Your text:", height=250)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Thinking..."):
            prompt = (
                "You are a psychologist trained in the Big Five and MBTI. "
                "Analyze the user's writing and return:\n"
                "1. Emotional tone\n"
                "2. Big Five personality trait estimates\n"
                "3. A likely MBTI type\n"
                "4. Personalized advice, book or career recommendations\n"
                "Be concise, accurate, and thoughtful.\n\n"
                f"User text: {user_text}\n"
            )

            try:
                output = client.text_generation(
                    prompt,
                    max_new_tokens=500,
                    temperature=0.7
                )
                st.subheader("📋 Results")
                st.markdown(output)
            except Exception as e:
                st.error(f"Unexpected error: {e}")

