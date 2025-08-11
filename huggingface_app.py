import os
import streamlit as st
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HF_TOKEN")  # set this in Streamlit secrets or env var
if not HF_TOKEN:
    st.error("Missing Hugging Face access token.")
    st.stop()

client = InferenceClient(model="openai-community/gpt-oss-20b", token=HF_TOKEN)

st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 Kath's Personality Analyzer - Hugging Face Model")

st.divider()

text = st.text_area("Paste your writing sample:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        prompt = (
            "You are a psychologist trained in the Big Five and MBTI. "
            "Analyze the user's writing and return:\n"
            "1. Emotional tone\n"
            "2. Big Five personality trait estimates\n"
            "3. A likely MBTI type\n"
            "4. Personalized advice, book or career recommendations"
        )
        with st.spinner("Analyzing..."):
            output = client.text_generation(
                prompt + "\nUser text: " + text,
                max_new_tokens=500,
                temperature=0.7
            )
        st.subheader("📋 Results")
        st.markdown(output)
