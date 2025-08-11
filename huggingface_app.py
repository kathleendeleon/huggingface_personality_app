import os
import streamlit as st
from openai import OpenAI
from openai import APIError, AuthenticationError, RateLimitError, NotFoundError

# Grab API key from Streamlit Cloud Secrets or env
api_key = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
if not api_key:
    st.error("🔐 Missing `HF_TOKEN`. Add it in Streamlit → Settings → Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)


st.set_page_config(page_title="Personality Analyzer", page_icon="🧠")
st.title("🧠 Kath's AI Personality Analyzer")
st.subheader("Hugging Face Model gpt-oss-20b")
st.write(
    "This app uses OpenAI’s **Responses API** with the open model **gpt-oss-20b**."
    " You’ll need a valid OpenAI API key with access.")


st.divider()

st.markdown("Paste your writing sample below and discover your personality profile based on tone, traits, and suggestions.")

# Optional: let you switch models quickly (e.g., use fallback if needed)
model_choice = st.selectbox(
    "Model",
    options=["gpt-oss-20b", "gpt-4o-mini"],
    index=0,
    help="If gpt-oss-20b isn’t enabled for your key, pick gpt-4o-mini."
)

user_text = st.text_area("✍️ Your text:", height=250)

if st.button("🔍 Analyze"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
        st.stop()

    system_prompt = (
        "You are a psychologist trained in the Big Five and MBTI. "
        "Analyze the user's writing and return:\n"
        "1) Emotional tone\n"
        "2) Big Five personality trait estimates (O,C,E,A,N)\n"
        "3) A likely MBTI type\n"
        "4) Personalized advice plus 2 book or career recommendations\n"
        "Be concise, accurate, and thoughtful."
    )

    def run_analysis(model_name: str) -> str:
        resp = client.responses.create(
            model=model_name,                      # <- Responses API model name
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ],
            temperature=0.7,
            max_output_tokens=700,                 # Responses API uses max_output_tokens
        )
        # Convenience property returns concatenated text
        return resp.output_text

    with st.spinner("Thinking..."):
        try:
            content = run_analysis(model_choice)
        except NotFoundError:
            # model not found for your account → auto-fallback
            if model_choice != "gpt-4o-mini":
                try:
                    content = run_analysis("gpt-4o-mini")
                    st.info("`gpt-oss-20b` not available for this key. Fell back to `gpt-4o-mini`.")
                except Exception as e:
                    st.error(f"API error after fallback: {e}")
                    st.stop()
            else:
                st.error("Selected model not found for this key.")
                st.stop()
        except AuthenticationError:
            st.error("🔑 Authentication failed. Check your `OPENAI_API_KEY`.")
            st.stop()
        except RateLimitError:
            st.error("🚫 Rate limit exceeded. Try again later or upgrade your plan.")
            st.stop()
        except APIError as e:
            st.error(f"API error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.subheader("📋 Results")
    st.markdown(content)
