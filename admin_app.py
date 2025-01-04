# admin_app.py

import os
import csv
import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from openai import OpenAI
import re

###############################################################################
# 0. Default/Pre-created Fine-Tuned Models
#    Each entry is a dict: {"id": <OpenAI model ID>, "public": bool}
###############################################################################
PRE_CREATED_MODELS = {
    "model_AlSpfqGn": {
        "id": "ft:gpt-3.5-turbo-0125:personal::AlSpfqGn",
        "public": True
    },
    "model_AlTffoN4": {
        "id": "ft:gpt-3.5-turbo-0125:personal::AlTffoN4",
        "public": False
    },
    "model_AlTewxyb": {
        "id": "ft:gpt-3.5-turbo-0125:personal::AlTewxyb",
        "public": True
    },
}

###############################################################################
# 1. Persisting Additional Models in JSON
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
                # Convert older string-based entries if needed
                if isinstance(data, dict):
                    converted = {}
                    for alias, val in data.items():
                        if isinstance(val, str):
                            # Old style: just "id" in string
                            converted[alias] = {"id": val, "public": True}
                        elif isinstance(val, dict):
                            if "id" not in val:
                                continue  # skip invalid entries
                            if "public" not in val:
                                val["public"] = True
                            converted[alias] = val
                    return converted
        except Exception as e:
            st.warning(f"Could not read {MODELS_JSON_FILE}: {e}")
    return {}

def save_saved_models(models_dict):
    """
    Save the given dictionary of alias -> {id, public} to local JSON file.
    """
    try:
        with open(MODELS_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(models_dict, f, indent=2)
    except Exception as e:
        st.error(f"Error saving models to {MODELS_JSON_FILE}: {e}")

###############################################################################
# 2. CSV Reading & JSONL Building
###############################################################################
def read_rows_from_multiple_csvs(uploaded_files):
    """
    Given multiple Streamlit-uploaded CSV files,
    read them all into a single list of row dicts.
    """
    all_rows = []
    for file_obj in uploaded_files:
        try:
            file_content = file_obj.getvalue().decode("utf-8", errors="replace")
            reader = csv.DictReader(file_content.splitlines())
            for row in reader:
                all_rows.append(row)
        except Exception as e:
            st.error(f"Error reading file {file_obj.name}: {e}")
    return all_rows

def build_jsonl_for_senders(all_rows, selected_senders, output_jsonl="filtered_data.jsonl"):
    """
    Filters rows to those matching ANY chosen senders, writes them to a chat-format JSONL.
    """
    count = 0
    lower_senders = [s.lower() for s in selected_senders]

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for row in all_rows:
            parsed_sender = (row.get("Parsed From") or "").strip().lower()
            if parsed_sender in lower_senders:
                subject = row.get("Parsed Subject", "").strip()
                body = row.get("Parsed Body", "").strip()

                user_content = f"Subject: {subject}\nPlease respond with the email body style."
                assistant_content = body

                data_line = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
                f_out.write(json.dumps(data_line) + "\n")
                count += 1
    return count

###############################################################################
# 3. Misc Utility
###############################################################################
def get_first_name(full_name: str) -> str:
    """Extract first name from a full name string."""
    return full_name.split()[0] if full_name else ""

###############################################################################
# 4. The Main Admin GUI
###############################################################################
def create_admin_gui():
    st.title("Admin - Fine-Tune & Test Models")

    # A) OpenAI API key handling
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        st.warning("No OPENAI_API_KEY found. Please enter it below or stop.")
        user_key = st.text_input("OpenAI API Key", type="password")
        if user_key:
            client = OpenAI(api_key=user_key)
        else:
            st.stop()
    else:
        client = OpenAI(api_key=openai_api_key)

    # B) Load or merge existing models
    if "fine_tuned_models" not in st.session_state:
        merged_models = dict(PRE_CREATED_MODELS)
        user_models = load_saved_models()
        for alias, data_dict in user_models.items():
            merged_models[alias] = data_dict
        st.session_state["fine_tuned_models"] = merged_models

    # Create three tabs
    tabs = st.tabs(["Test Models", "Manage Models", "Fine-Tune a New Model"])

    ###########################################################################
    # TAB 1: TEST MODELS
    ###########################################################################
    with tabs[0]:
        st.subheader("Test an Existing Model")

        all_models = st.session_state["fine_tuned_models"]
        if not all_models:
            st.info("No models are available yet. Add or fine-tune a model in another tab.")
        else:
            aliases = list(all_models.keys())
            selected_alias = st.selectbox("Select a Model", aliases)
            chosen_data = all_models[selected_alias]
            chosen_model_id = chosen_data["id"]

            system_prompt = st.text_area("System Prompt (optional)", value="", height=80)
            temp_val = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

            test_tabs = st.tabs(["Rewrite Email", "Modify Reply", "Generate Reply"])

            # 1) Rewrite
            with test_tabs[0]:
                email_to_rewrite = st.text_area(
                    "Enter email to rewrite:",
                    height=200,
                    key="test_models_rewrite_email"
                )
                if st.button("Generate Rewrite", key="rewrite_button"):
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
                                model=chosen_model_id,
                                messages=messages,
                                temperature=temp_val
                            )
                            txt = response.choices[0].message.content
                            st.write("**Rewritten Email:**")
                            st.write(txt)
                        except Exception as e:
                            st.error(f"Error generating response: {e}")

            # 2) Modify Reply
            with test_tabs[1]:
                col1, col2 = st.columns(2)
                with col1:
                    original_email_mod = st.text_area(
                        "Original Email:", height=200, key="test_models_modify_original"
                    )
                with col2:
                    my_reply_mod = st.text_area(
                        "Your Reply:", height=200, key="test_models_modify_reply"
                    )

                if st.button("Generate Modified Reply", key="modify_reply_button"):
                    with st.spinner("Generating..."):
                        try:
                            if not my_reply_mod.strip():
                                prompt = f"Please rewrite this email:\n\n{original_email_mod}"
                            else:
                                prompt = (
                                    f"Original Email:\n{original_email_mod}\n\n"
                                    f"Your Reply:\n{my_reply_mod}\n\n"
                                    f"Please improve this reply while maintaining the same general message."
                                )

                            messages = []
                            if system_prompt.strip():
                                messages.append({"role": "system", "content": system_prompt})
                            messages.append({"role": "user", "content": prompt})

                            response = client.chat.completions.create(
                                model=chosen_model_id,
                                messages=messages,
                                temperature=temp_val
                            )
                            txt = response.choices[0].message.content
                            st.write("**Modified Reply:**" if my_reply_mod.strip() else "**Rewritten Email:**")
                            st.write(txt)
                        except Exception as e:
                            st.error(f"Error generating response: {e}")

            # 3) Generate Reply
            with test_tabs[2]:
                original_email_gen = st.text_area(
                    "Original Email:", height=200, key="test_models_generate_original"
                )
                if st.button("Generate Reply", key="generate_reply_button"):
                    with st.spinner("Generating..."):
                        try:
                            messages = []
                            if system_prompt.strip():
                                messages.append({"role": "system", "content": system_prompt})
                            messages.append({
                                "role": "user",
                                "content": f"Please write a reply to this email:\n\n{original_email_gen}"
                            })

                            response = client.chat.completions.create(
                                model=chosen_model_id,
                                messages=messages,
                                temperature=temp_val
                            )
                            txt = response.choices[0].message.content
                            st.write("**Generated Reply:**")
                            st.write(txt)
                        except Exception as e:
                            st.error(f"Error generating response: {e}")

    ###########################################################################
    # TAB 2: MANAGE MODELS
    ###########################################################################
    with tabs[1]:
        st.subheader("View, Edit, Remove, or Add Models")

        """
        **Existing Models**:
        
        - You can update the **Model ID** or toggle whether it's **Public**.
        - Check "Remove this model?" to delete it from the list entirely.
        
        When done, click **"Save All Changes"** at the bottom.
        """

        all_models = st.session_state["fine_tuned_models"]
        if not all_models:
            st.info("No models in memory yet. Fine-tune or add a new model below.")
        else:
            updated_data = {}
            removed_aliases = []

            # Display each model in an expander
            for alias in sorted(all_models.keys()):
                model_info = all_models[alias]
                with st.expander(f"Alias: {alias}", expanded=False):
                    new_id = st.text_input(
                        "Model ID",
                        value=model_info["id"],
                        key=f"model_id_{alias}"
                    )
                    new_public = st.checkbox(
                        "Public?",
                        value=model_info.get("public", True),
                        key=f"public_{alias}"
                    )
                    remove_model = st.checkbox(
                        "Remove this model?",
                        value=False,
                        key=f"remove_{alias}"
                    )

                    # We'll store these in a dict for later saving
                    updated_data[alias] = {
                        "id": new_id,
                        "public": new_public
                    }
                    if remove_model:
                        removed_aliases.append(alias)

            st.write("---")
            if st.button("Save All Changes"):
                # Apply the updated data to session
                for alias, val in updated_data.items():
                    if alias not in removed_aliases:
                        st.session_state["fine_tuned_models"][alias] = val

                # Remove any aliases that are flagged
                for r_alias in removed_aliases:
                    if r_alias in st.session_state["fine_tuned_models"]:
                        del st.session_state["fine_tuned_models"][r_alias]

                # Save to disk
                user_models = load_saved_models()  # current disk data
                # Overwrite or remove as needed
                for alias, val in updated_data.items():
                    if alias not in removed_aliases:
                        user_models[alias] = val
                for r_alias in removed_aliases:
                    if r_alias in user_models:
                        del user_models[r_alias]

                save_saved_models(user_models)
                st.success("All changes saved!")

        st.markdown("---")
        st.markdown("### Add a New (or Existing) Model")
        """
        Enter an **alias** (if it already exists, you'll **override** that model).
        Enter the **model ID** (like `ft:gpt-3.5-turbo-0125:...`).
        Choose whether it's **public**. 
        Then click **"Add/Update Model."**
        """
        alias_in = st.text_input("Alias for your model (e.g., 'my_new_model')", "")
        id_in = st.text_input("Full Model ID (e.g., 'ft:gpt-3.5-turbo-0125:...')", "")
        public_in = st.checkbox("Public?", value=True)

        if st.button("Add/Update Model"):
            if alias_in.strip() and id_in.strip():
                # Update session
                st.session_state["fine_tuned_models"][alias_in] = {
                    "id": id_in,
                    "public": public_in
                }
                # Update disk
                user_models = load_saved_models()
                user_models[alias_in] = {"id": id_in, "public": public_in}
                save_saved_models(user_models)

                st.success(f"Model '{alias_in}' added/updated successfully!")
            else:
                st.warning("Please provide both an alias and a model ID.")

    ###########################################################################
    # TAB 3: FINE-TUNE A NEW MODEL
    ###########################################################################
    with tabs[2]:
        st.subheader("Upload CSV & Fine-Tune a Model")

        csv_files = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True
        )

        if csv_files:
            all_rows = read_rows_from_multiple_csvs(csv_files)
            if not all_rows:
                st.warning("No data read from CSV(s). Check your file format.")
            else:
                # Gather unique senders
                senders = set()
                for row in all_rows:
                    frm = (row.get("Parsed From") or "").strip()
                    if frm:
                        senders.add(frm)
                senders = sorted(senders)

                group_by_first_name = st.checkbox("Group senders by first name", value=False)
                if group_by_first_name:
                    grouped_senders = {}
                    for sender in senders:
                        first_name = get_first_name(sender)
                        if first_name not in grouped_senders:
                            grouped_senders[first_name] = []
                        grouped_senders[first_name].append(sender)

                    display_senders = []
                    for fn, full_list in grouped_senders.items():
                        if len(full_list) > 1:
                            display_senders.append(f"{fn} (All variations)")
                        else:
                            display_senders.append(full_list[0])

                    selected_disp = st.multiselect("Select Sender(s) to Filter On", sorted(display_senders))
                    selected_senders = []
                    for d in selected_disp:
                        if d.endswith("(All variations)"):
                            fn = d.split()[0]
                            selected_senders.extend(grouped_senders[fn])
                        else:
                            selected_senders.append(d)
                else:
                    selected_senders = st.multiselect("Select Sender(s) to Filter On", senders)

                if selected_senders:
                    st.write("Selected Sender(s):", selected_senders)
                    if st.button("Generate JSONL for Selected Senders"):
                        jsonl_file_name = "filtered_data.jsonl"
                        count = build_jsonl_for_senders(all_rows, selected_senders, jsonl_file_name)
                        if count > 0:
                            st.success(f"Created {jsonl_file_name} with {count} examples.")
                            st.session_state["jsonl_file"] = jsonl_file_name
                        else:
                            st.warning("No rows found for those senders.")
                else:
                    st.info("Pick at least one sender to build a JSONL for fine-tuning.")
        else:
            st.info("Upload CSV(s) to create a new fine-tuned model.")

        st.markdown("---")

        # Fine-tune parameters
        st.subheader("Fine-Tune Parameters")
        base_model = st.selectbox("Base Model", ["gpt-3.5-turbo", "gpt-4"])
        n_epochs = st.number_input("Number of epochs", min_value=1, value=1, step=1)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=8, step=1)
        learning_rate = st.number_input("Learning Rate Multiplier", min_value=0.01, max_value=1.0, value=0.05, step=0.01)

        if st.button("Start Fine-Tuning Job"):
            if "jsonl_file" not in st.session_state:
                st.warning("No JSONL file is available. Generate it above first.")
            else:
                file_path = st.session_state["jsonl_file"]
                try:
                    with open(file_path, "rb") as f:
                        resp = client.files.create(file=f, purpose="fine-tune")
                    file_id = resp.id
                    st.write(f"Uploaded training file. File ID = {file_id}")
                except Exception as e:
                    st.error(f"Error uploading training file: {e}")
                    return

                try:
                    job = client.fine_tuning.jobs.create(
                        training_file=file_id,
                        model=base_model,
                        hyperparameters={
                            "n_epochs": n_epochs,
                            "batch_size": batch_size,
                            "learning_rate_multiplier": learning_rate
                        }
                    )
                    job_id = job.id
                    st.success(f"Fine-tune job created! Job ID: {job_id}")
                    st.session_state["current_finetune_job"] = job_id
                    st.session_state["current_base_model"] = base_model
                except Exception as e:
                    st.error(f"Error creating fine-tune job: {e}")
                    return

        st.markdown("---")

        # Monitor fine-tune job
        st.subheader("Monitor Fine-Tune Job")
        enable_auto_refresh = st.checkbox("Enable auto-refresh", value=True)

        if "current_finetune_job" in st.session_state:
            job_id = st.session_state["current_finetune_job"]
            st.write(f"Current Fine-Tune Job ID: {job_id}")

            try:
                if enable_auto_refresh:
                    count_refreshed = st_autorefresh(interval=10_000, limit=1000, key="ft_auto_refresh")
                    st.write(f"**Job Status (auto-refreshed {count_refreshed} times)**:")
                else:
                    st.write("**Job Status**:")

                status = client.fine_tuning.jobs.retrieve(job_id)
                st.json(status.to_dict())

                if status.status == "succeeded":
                    st.success("Fine-tune succeeded!")
                    ft_model = status.fine_tuned_model
                    if not ft_model:
                        st.warning("No 'fine_tuned_model' found in response.")
                    else:
                        new_key = f"{ft_model}-{len(st.session_state['fine_tuned_models'])+1}"
                        # Newly fine-tuned models default to public = True
                        st.session_state["fine_tuned_models"][new_key] = {"id": ft_model, "public": True}

                        user_models = load_saved_models()
                        user_models[new_key] = {"id": ft_model, "public": True}
                        save_saved_models(user_models)

                        st.write(f"Fine-tuned model: `{ft_model}` stored in session & disk.")
                elif status.status == "failed":
                    st.error("Fine-tune failed.")
            except Exception as e:
                st.error(f"Error retrieving job status: {e}")
        else:
            st.info("No active fine-tune job to monitor.")

def main():
    st.set_page_config(page_title="Admin - Fine-Tune & Test Models", layout="wide")
    create_admin_gui()

if __name__ == "__main__":
    main()
