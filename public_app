import os
import json
import streamlit as st
from openai import OpenAI
from datetime import datetime
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="missing ScriptRunContext")

# Set bare mode
st._is_running_with_streamlit = False

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
    # st.markdown("## Email Tools (Public)")

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

    # Automatically select the top two models by alphabetical order
    model_keys = sorted(public_models.keys())
    if len(model_keys) < 2:
        st.warning("Not enough public models available.")
        return
    chosen_models = [public_models[key] for key in model_keys[:2]]

    # Input fields for Name and Location
    sender_name = st.text_input("Name", value="")
    sender_location = st.text_input("Location", value="")

    # Modify the system prompt to include Name and Location, with default location as New York
    if not sender_location.strip():
        sender_location = "New York"
    system_prompt = f"Sender Name: {sender_name}, Location: {sender_location}"

    # Add explicit instruction to avoid specific names unless they appear in inputs
    protected_names = ['Jos', 'Scott', 'Lesley', 'Diaz']
    if not any(name in sender_name for name in protected_names):
        system_prompt += "\nIMPORTANT: NEVER use the names Jos, Scott, Lesley, or Diaz in your responses."

    # Remove the temperature slider
    # temp_val = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    # Remove the 'Email Rewrite / Reply Tools' header
    # st.markdown("---")
    # st.markdown("### Email Rewrite / Reply Tools")

    # Tabs for rewriting, modifying, and generating replies
    tabs = st.tabs(["Rewrite Email", "Generate Reply", "Modify Reply"])

    def is_response_clean(response_text, protected_names, input_text):
        """Check if response contains any protected names that weren't in the input"""
        for name in protected_names:
            if name.lower() in response_text.lower() and name.lower() not in input_text.lower():
                return False
        return True

    def get_clean_response(client, model, messages, temp_val, protected_names, input_text, max_attempts=3):
        """Get a response that doesn't contain protected names (unless they were in input)"""
        for attempt in range(max_attempts):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp_val
            )
            response_text = response.choices[0].message.content
            # Remove 'Santiago' and 'Santi' from the response
            response_text = response_text.replace('Santiago', '').replace('Santi', '')
            if is_response_clean(response_text, protected_names, input_text):
                return response_text
            
            # On last attempt, replace protected names with [Name]
            if attempt == max_attempts - 1:
                modified_text = response_text
                for name in protected_names:
                    if name.lower() in modified_text.lower() and name.lower() not in input_text.lower():
                        modified_text = modified_text.replace(name, '[Name]')
                        # Also handle case variations
                        modified_text = modified_text.replace(name.lower(), '[Name]')
                        modified_text = modified_text.replace(name.upper(), '[Name]')
                        modified_text = modified_text.replace(name.capitalize(), '[Name]')
                return modified_text
        return None  # This should never be reached now

    # 1) Rewrite Email
    with tabs[0]:
        email_to_rewrite = st.text_area("Enter email to rewrite", height=200)
        if st.button("Rewrite"):
            with st.spinner("Generating..."):
                try:
                    messages = []
                    if system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({
                        "role": "user",
                        "content": f"Please rewrite this email:\n\n{email_to_rewrite}"
                    })

                    responses = {alias: [] for alias in public_models.keys()}
                    for alias, model in public_models.items():
                        num_responses = 3 if model == chosen_models[0] else 1
                        for _ in range(num_responses):
                            # Combine all input text for checking
                            input_text = f"{sender_name} {email_to_rewrite}".lower()
                            clean_response = get_clean_response(
                                client, model, messages, 0.7, 
                                protected_names, input_text
                            )
                            if clean_response:
                                responses[alias].append(clean_response)
                            else:
                                st.warning(f"Could not generate appropriate response after multiple attempts")

                    for alias, model_responses in responses.items():
                        response_container = st.container()
                        with response_container:
                            for i, txt in enumerate(model_responses, 1):
                                col1, col2 = st.columns([20, 1])
                                with col1:
                                    st.text_area("", value=txt, height=150, key=f"response_{alias}_{i}", label_visibility="collapsed")
                                with col2:
                                    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)  # Add some spacing
                                st.markdown(f"<div style='font-size: 10px; color: #888; text-align: right;'>{alias}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # 2) Generate Reply
    with tabs[1]:
        original_email = st.text_area("Original Email", height=200, key="generate_original")
        if st.button("Reply"):
            with st.spinner("Generating..."):
                try:
                    messages = []
                    if system_prompt.strip():
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({
                        "role": "user",
                        "content": f"Please write a reply to this email:\n\n{original_email}"
                    })

                    responses = {alias: [] for alias in public_models.keys()}
                    for alias, model in public_models.items():
                        num_responses = 1 if model == chosen_models[0] else 3
                        for _ in range(num_responses):
                            # Combine all input text for checking
                            input_text = f"{sender_name} {original_email}".lower()
                            clean_response = get_clean_response(
                                client, model, messages, 0.7, 
                                protected_names, input_text
                            )
                            if clean_response:
                                responses[alias].append(clean_response)
                            else:
                                st.warning(f"Could not generate appropriate response after multiple attempts")

                    for alias, model_responses in responses.items():
                        response_container = st.container()
                        with response_container:
                            for i, txt in enumerate(model_responses, 1):
                                col1, col2 = st.columns([20, 1])
                                with col1:
                                    st.text_area("", value=txt, height=150, key=f"response_{alias}_{i}", label_visibility="collapsed")
                                with col2:
                                    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)  # Add some spacing
                                st.markdown(f"<div style='font-size: 10px; color: #888; text-align: right;'>{alias}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # 3) Modify Reply
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            original_email = st.text_area("Original Email", height=200, key="modify_original")
        with col2:
            my_reply = st.text_area("Your Reply", height=200, key="modify_reply")

        if st.button("Modify"):
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

                    responses = {alias: [] for alias in public_models.keys()}
                    for alias, model in public_models.items():
                        num_responses = 1 if model == chosen_models[0] else 3
                        for _ in range(num_responses):
                            # Combine all input text for checking
                            input_text = f"{sender_name} {original_email} {my_reply}".lower()
                            clean_response = get_clean_response(
                                client, model, messages, 0.7, 
                                protected_names, input_text
                            )
                            if clean_response:
                                responses[alias].append(clean_response)
                            else:
                                st.warning(f"Could not generate appropriate response after multiple attempts")

                    for alias, model_responses in responses.items():
                        response_container = st.container()
                        with response_container:
                            for i, txt in enumerate(model_responses, 1):
                                col1, col2 = st.columns([20, 1])
                                with col1:
                                    st.text_area("", value=txt, height=150, key=f"response_{alias}_{i}", label_visibility="collapsed")
                                with col2:
                                    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)  # Add some spacing
                                st.markdown(f"<div style='font-size: 10px; color: #888; text-align: right;'>{alias}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Add timestamp in the top right corner
    now = datetime.now()
    timestamp = now.strftime('%b %d %I:%M %p')
    st.markdown(
        f"""
        <div style='position: absolute; top: 10px; right: 10px; font-size: 12px; color: #888;'>
            {timestamp}
        </div>
        """,
        unsafe_allow_html=True
    )

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

    # Set a timeout period (in seconds)
    TIMEOUT = 300  # 5 minutes

    def check_inactivity():
        if 'last_active' not in st.session_state:
            st.session_state.last_active = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_active > TIMEOUT:
            st.warning("Session timed out due to inactivity.")
            # Stop the server
            os._exit(0)

    # Call this function at the start of your app
    check_inactivity()

    create_public_gui()

if __name__ == "__main__":
    main()
