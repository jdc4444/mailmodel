import os
import json
import streamlit as st
from openai import OpenAI

###############################################################################
# 1. Load user-saved models only
###############################################################################
MODELS_JSON_FILE = "my_saved_models.json"

def load_saved_models():
    """
    Load previously saved models from local JSON file, or return an empty dict.
    Each model is stored as: alias -> {"id": <str>, "public": <bool>}
    """
    if os.path.exists(MODELS_JSON_FILE):
        try:
            with open(MODELS_JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            st.warning(f"Could not read {MODELS_JSON_FILE}: {e}")
    return {}

###############################################################################
# 2. Minimalist Swiss-Style Public GUI
###############################################################################
def create_public_gui():
    # Minimal heading instead of big title
    st.markdown("## Email Tools (Public)")

    # A) OpenAI API key handling
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.error("No OPENAI_API_KEY found on the server. Please configure it or enter manually.")
        user_key = st.text_input("OpenAI API Key", type="password")
        if not user_key:
            return
        client = OpenAI(api_key=user_key)
    else:
        client = OpenAI(api_key=openai_api_key)

    # B) Load models from disk
    all_models = load_saved_models()

    # Filter out models where "public" is False
    public_models = {}
    for alias, model_info in all_models.items():
        # If the model entry is a dict with 'id' and 'public' keys
        if isinstance(model_info, dict):
            # Only show if public == True
            if model_info.get("public", True):
                public_models[alias] = model_info["id"]
        # If it's an older string-based entry, we assume it's public
        elif isinstance(model_info, str):
            public_models[alias] = model_info

    if not public_models:
        st.warning("No publicly visible models. Please contact the admin or make some models public.")
        return

    # Let the user pick one of the PUBLIC models
    model_keys = sorted(public_models.keys())
    selected_model_key = st.selectbox("Select a Model", model_keys)
    chosen_model = public_models[selected_model_key]

    # Optional system prompt & temperature
    system_prompt = st.text_area("System Prompt (optional)", value="", height=80)
    temp_val = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    st.markdown("---")
    st.markdown("### Email Rewrite / Reply Tools")

    # Tabs for rewriting, modifying, and generating replies
    tabs = st.tabs(["Rewrite Email", "Modify Reply", "Generate Reply"])

    # 1) Rewrite Email
    with tabs[0]:
        email_to_rewrite = st.text_area("Enter email to rewrite", height=200)
        if st.button("Generate Rewrite"):
            with st.spinner("Generating..."):
                try:
                    messages = []
                    if system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({
                        "role": "user",
                        "content": f"Please rewrite this email:\n\n{email_to_rewrite}"
                    })

                    response = client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=temp_val
                    )
                    txt = response.choices[0].message.content
                    st.write("**Rewritten Email:**")
                    st.write(txt)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # 2) Modify Reply
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            original_email = st.text_area("Original Email", height=200, key="modify_original")
        with col2:
            my_reply = st.text_area("Your Reply", height=200, key="modify_reply")

        if st.button("Generate Modified Reply"):
            with st.spinner("Generating..."):
                try:
                    if not my_reply.strip():
                        prompt = f"Please rewrite this email:\n\n{original_email}"
                    else:
                        prompt = (
                            f"Original Email:\n{original_email}\n\n"
                            f"Your Reply:\n{my_reply}\n\n"
                            f"Please improve this reply while maintaining the same general message."
                        )

                    messages = []
                    if system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    response = client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=temp_val
                    )
                    txt = response.choices[0].message.content
                    st.write("**Modified Reply:**" if my_reply.strip() else "**Rewritten Email:**")
                    st.write(txt)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # 3) Generate Reply
    with tabs[2]:
        original_email = st.text_area("Original Email", height=200, key="generate_original")
        if st.button("Generate Reply"):
            with st.spinner("Generating..."):
                try:
                    messages = []
                    if system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({
                        "role": "user",
                        "content": f"Please write a reply to this email:\n\n{original_email}"
                    })

                    response = client.chat.completions.create(
                        model=chosen_model,
                        messages=messages,
                        temperature=temp_val
                    )
                    txt = response.choices[0].message.content
                    st.write("**Generated Reply:**")
                    st.write(txt)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

###############################################################################
# 3. Main entrypoint, with minimalist theming
###############################################################################
def main():
    # Set page config
    st.set_page_config(page_title="Email Tools (Public)", layout="centered", page_icon="✉️")
    
    # Automatically switch based on system's dark mode setting
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: dark) {
            /* Dark mode styles */
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #111 !important;
                color: #fff !important;
            }
            textarea, input[type="text"], input[type="password"] {
                border: 1px solid #444 !important;
                background-color: #222 !important;
                color: #fff !important;
            }
            .stButton>button {
                background-color: #444 !important;
                color: #fff !important;
            }
            .stButton>button:hover {
                background-color: #666 !important;
            }
            div[data-baseweb="tab"] > button {
                background-color: #333 !important;
                color: #fff !important;
            }
            div[data-baseweb="tab"] > button[aria-selected="true"] {
                background-color: #555 !important;
                color: #fff !important;
            }
        }
        @media (prefers-color-scheme: light) {
            /* Light mode styles */
            html, body, [data-testid="stAppViewContainer"] {
                background-color: #ffffff !important;
                color: #111 !important;
            }
            textarea, input[type="text"], input[type="password"] {
                border: 1px solid #ccc !important;
                background-color: #fff !important;
                color: #111 !important;
            }
            .stButton>button {
                background-color: #111 !important;
                color: #fff !important;
            }
            .stButton>button:hover {
                background-color: #333 !important;
            }
            div[data-baseweb="tab"] > button {
                background-color: #f0f0f0 !important;
                color: #111 !important;
            }
            div[data-baseweb="tab"] > button[aria-selected="true"] {
                background-color: #111 !important;
                color: #fff !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    create_public_gui()

if __name__ == "__main__":
    main()
