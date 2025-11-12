import requests
import time
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
import os
import re
import glob  
import streamlit as st

# --- Configuration (Moved some to Streamlit UI) ---

# 3. The Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Data dir for persistent data
DATA_DIR = "app_data"

# 4. The Database files
DB_FILE = os.path.join(DATA_DIR, "debate.db")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_store")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
VOTETRANSCRIPT_DIR = os.path.join(TRANSCRIPT_DIR, "vote_transcripts")

VECTOR_COLLECTION_NAME = "debate_history"

# --- Set up the Streamlit page ---
st.set_page_config(
    page_title="Local AI Persona Panel",
    page_icon="🎤",
    layout="wide"
)

# st.write("Folder Watch Blacklist:", st.get_option("server.folderWatchBlacklist"))

# 1. Define your "debaters"
if 'PANELISTS' not in st.session_state:
    st.session_state.PANELISTS = {
        "Modern Liberal": "phi3:medium",
        "Modern Conservative": "phi3:medium",
        "Libertarian": "phi3:medium"
}
    
if 'PERSONA_DESCRIPTIONS' not in st.session_state:
    st.session_state.PERSONA_DESCRIPTIONS = {
        "Modern Liberal": """You are a Modern Liberal. You believe that government has a moral 
        obligation to care for its citizens and address systemic inequalities. 
        Your analysis must prioritize fairness, community well-being, and 
        environmental protection. You see collective action, led by the 
        government, as the best way to achieve a just society.""",
    
        "Modern Conservative": """You are a Modern Conservative. You believe in individual liberty, 
        personal responsibility, and the power of the free market. Your analysis 
        must prioritize limited government, economic freedom, and a strong rule of 
        law. You believe that the best solutions come from individual actors 
        and private institutions, not government bureaucracy.""",
        
        "Libertarian": """You are a Libertarian. Your single, core belief is in maximum 
        individual freedom. Your analysis must oppose any government 
        intervention—economic or social—that is not absolutely necessary 
        to protect the life, liberty, and property of individuals. You are 
        skeptical of both the Liberal and Conservative positions, as you see 
        both as attempts to use government power to control people."""
}
if 'transcript_view' not in st.session_state:
    st.session_state.transcript_view = "Full Transcripts"
if 'full_log_files' not in st.session_state:
    st.session_state.full_log_files = sorted(glob.glob(f"{TRANSCRIPT_DIR}/debate_*.txt"), reverse=True)
if 'voting_log_files' not in st.session_state:
    st.session_state.voting_log_files = sorted(glob.glob(f"{TRANSCRIPT_DIR}/voting_*.txt"), reverse=True)
if 'log_content_cache' not in st.session_state:
    st.session_state.log_content_cache = {"file": None, "content": None}

# --- NEW: Cache the models and DB connection ---
# @st.cache_resource is a decorator that tells Streamlit to run
# this function *once* and keep the result in memory.
# This prevents reloading the embedding model every time we click a button.
@st.cache_resource
def get_embedding_model():
    print("[Cache Miss] Loading embedding model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[Embedding model loaded.]")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def get_chroma_client():
    print("[Cache Miss] Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_or_create_collection(name=VECTOR_COLLECTION_NAME)
        print(f"[Vector database connected at '{VECTOR_DB_PATH}'.]")
        return collection
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")
        return None
    
@st.cache_resource
def init_db():
    print("[Cache Miss] Init DB...")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS debates (
        debate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        winning_persona TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS arguments (
        argument_id INTEGER PRIMARY KEY AUTOINCREMENT,
        debate_id INTEGER NOT NULL,
        persona TEXT NOT NULL,
        model TEXT NOT NULL,
        phase TEXT NOT NULL,
        argument_text TEXT NOT NULL,
        FOREIGN KEY (debate_id) REFERENCES debates (debate_id)
        )
    ''')
    conn.commit()
    conn.close()
    st.info("SQL database and Vector database are ready.")

# Load the models
EMBEDDING_MODEL = get_embedding_model()

# A "Safer" Learning Function ---
def get_historical_context(persona, question):
    """
    Queries the ChromaDB vector store for semantically similar
    past arguments. This *new* version creates a "safer" prompt
    that is less likely to confuse the models.
    """
    st.info(f"[Searching vector database for relevant history for {persona}...] \n")
    CHROMA_COLLECTION = get_chroma_client()
    
    try:
        # 1. Convert the new question into a vector
        question_embedding = EMBEDDING_MODEL.encode(question).tolist()
        # 2. Query ChromaDB
        # Find the 2 most similar arguments this persona has made in the past.
        results = CHROMA_COLLECTION.query(
            query_embeddings=[question_embedding],
            n_results=2,
            where={"persona": persona} # Only find arguments from this persona
        )
        history = ""
        win_count = 0
        loss_count = 0
        if results['documents'] and results['metadatas']:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                if meta.get('is_winner'):
                    win_count += 1
                else:
                    loss_count += 1
        
        # --- THIS IS THE KEY ---
        # Instead of returning the full, "poisonous" text,
        # we return a simple, one-line summary.
        if win_count > 0 and loss_count > 0:
            history = "In the past, your arguments on this topic have had mixed results. Be persuasive."
        elif win_count > 0:
            history = "Your past arguments on this topic have been very successful. Keep it up."
        elif loss_count > 0:
            history = "Your past arguments on this topic have lost. You must change your strategy to be more persuasive."

        if history:
            return "--- RELEVANT HISTORICAL CONTEXT ---\n" + history + "\n--- END CONTEXT ---\n"
        else:
            return "No relevant history found for this specific topic. Good luck.\n"
            
    except Exception as e:
        st.error(f"Error querying vector database: {e}")
        return "Error reading vector history. Relying on base logic.\n"

def log_debate_to_db(question, winning_persona, arguments_log):
    init_db()
    st.info("[Logging debate results to SQL and Vector databases...]")
    CHROMA_COLLECTION = get_chroma_client()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO debates (question, winning_persona) VALUES (?, ?)", (question, winning_persona))
        debate_id = cursor.lastrowid
        chroma_ids = []
        chroma_documents = []
        chroma_metadatas = []
        for persona, phases in arguments_log.items():
            if "rebuttal" in phases:
                model = st.session_state.PANELISTS[persona]
                argument_text = phases["rebuttal"]
                cursor.execute('''
                INSERT INTO arguments (debate_id, persona, model, phase, argument_text)
                VALUES (?, ?, ?, ?, ?)
                ''', (debate_id, persona, model, "rebuttal", argument_text))
                argument_id = cursor.lastrowid
                chroma_ids.append(f"arg_{argument_id}")
                chroma_documents.append(argument_text)
                chroma_metadatas.append({
                    "persona": persona,
                    "debate_id": debate_id,
                    "is_winner": (persona == winning_persona)
                })
        conn.commit()
        conn.close()
        st.success("Debate successfully logged to SQL database.")
        
        if chroma_documents:
            embeddings = EMBEDDING_MODEL.encode(chroma_documents).tolist()
            CHROMA_COLLECTION.add(
                embeddings=embeddings,
                documents=chroma_documents,
                metadatas=chroma_metadatas,
                ids=chroma_ids
            )
            st.success("Arguments successfully embedded and logged to Vector database.")

    except Exception as e:
        st.error(f"Error logging to databases: {e}")


def get_model_response(model_name, prompt):
    # This function now writes to the UI *before* the call
    with st.spinner(f"Querying {model_name} for its thoughts..."):
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        try:
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response").strip()
        except requests.exceptions.Timeout:
            st.error(f"Error: The request to {model_name} timed out.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Ollama API for {model_name}: {e}")
            return None


def save_transcript_to_file(transcript_log, user_question):
    # This function is now passed the question
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_question = re.sub(r'[^\w\s]', '', user_question.lower())
    safe_question_snippet = "_".join(safe_question.split()[:5])
    base_filename = f"debate_{timestamp}_{safe_question_snippet}.txt"
    filename = os.path.join(TRANSCRIPT_DIR, base_filename)
    st.info(f"Saving full transcript to: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(transcript_log))
        st.success("[Transcript saved successfully.]")
        st.session_state.full_log_files = sorted(glob.glob(f"{TRANSCRIPT_DIR}/debate_*.txt"), reverse=True)
    except IOError as e:
        st.error(f"Error saving transcript file: {e}")

def update_config():
    # --- Panelist Configuration ---
    with st.expander("⚙️ Manage Existing Panelist"):
        st.subheader("Manage Debaters and Personas")

        current_panelists = list(st.session_state.PANELISTS.keys())
        if not current_panelists:
            st.info("No panelists configured yet. Add one below!")

        # Each panelist gets their own form to prevent conflicts
        for persona_name in current_panelists:
            with st.form(key=f"form_for_{persona_name}"):
                st.markdown(f"#### Edit: {persona_name}")
                
                # --- WIDGETS ---
                # These just *display* the current value
                new_persona_name = st.text_input(
                    "Panelist Name", 
                    value=persona_name, 
                    key=f"name_{persona_name}"
                )
                model_name = st.text_input(
                    "Ollama Model", 
                    value=st.session_state.PANELISTS[persona_name], 
                    key=f"model_{persona_name}"
                )
                description = st.text_area(
                    "Description",
                    value=st.session_state.PERSONA_DESCRIPTIONS.get(persona_name, ""),
                    key=f"desc_{persona_name}",
                    height=150
                )
                
                # --- ACTION BUTTONS ---
                col1, col2 = st.columns([1, 1])
                with col1:
                    save_button = st.form_submit_button("Save Changes")
                with col2:
                    remove_button = st.form_submit_button(f"Remove {persona_name}", type="secondary")

                # --- LOGIC ---
                # This logic now *only* runs when a button is clicked
                if save_button:
                    # 1. Update the simple values
                    st.session_state.PANELISTS[persona_name] = model_name
                    st.session_state.PERSONA_DESCRIPTIONS[persona_name] = description

                    st.info(f"[Updating panelist '{persona_name}'...]")
                    # 2. Handle re-naming
                    if new_persona_name != persona_name:
                        # Copy data to new name
                        st.session_state.PANELISTS[new_persona_name] = st.session_state.PANELISTS.pop(persona_name)
                        st.session_state.PERSONA_DESCRIPTIONS[new_persona_name] = st.session_state.PERSONA_DESCRIPTIONS.pop(persona_name)
                        st.info(f"[Renaming panelist '{persona_name}' to '{new_persona_name}'...]")
                    
                    st.success(f"Panelist '{new_persona_name}' saved!")
                    st.rerun() # Re-run to update the form keys

                if remove_button:
                    st.info(f"[Removing panelist '{persona_name}'...]")
                    # Delete the panelist
                    del st.session_state.PANELISTS[persona_name]
                    if persona_name in st.session_state.PERSONA_DESCRIPTIONS:
                        del st.session_state.PERSONA_DESCRIPTIONS[persona_name]
                    
                    st.warning(f"Panelist '{persona_name}' removed.")
                    st.rerun() # Re-run to remove the form

def save_vote_transcript(voting_log, user_question):
    """
    Saves only the voting phase and final results to a separate transcript file.
    """
    os.makedirs(VOTETRANSCRIPT_DIR, exist_ok=True)
    now= datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    safe_question = re.sub(r'[^\w\s]', '', user_question.lower())
    safe_question_snippet = "_".join(safe_question.split()[:5])

    base_filename = f"vote_transcript_{timestamp}_{safe_question_snippet}.txt"
    filename = os.path.join(VOTETRANSCRIPT_DIR, base_filename)

    st.info(f"Saving vote transcript to: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(voting_log))
        st.success("[Vote transcript saved successfully.]")
        st.session_state.voting_log_files = sorted(glob.glob(f"{VOTETRANSCRIPT_DIR}/vote_transcript_*.txt"), reverse=True)
    except IOError as e:
        st.error(f"Error saving vote transcript file: {e}")

# --- Main Debate Logic (Modified for Streamlit) ---

def run_debate(user_question):
    """
    Orchestrates the entire 3-phase debate and saves a transcript.
    This function now uses st.write/st.markdown to print.
    """
    
    # --- List to store the full transcript ---
    transcript_log = []
    voting_log = []

    st.markdown("---")
    st.header("🎙️ Persona Panel Has Convened!")
    st.markdown("---")
    transcript_log.append("==============================================" \
                          "\n    Ethical Debate Panel Has Convened!     \n" \
                          "==============================================")
    
    question_line = f"\nTonight's question: {user_question}\n"
    st.subheader(f"Tonight's question: {user_question}")
    transcript_log.append(question_line)

    st.markdown("---")
    st.subheader("Phase 1: Opening Statements")
    transcript_log.append("\nPhase 1: Opening Statements\n" + "-"*30)
    
    arguments_log = {persona: {} for persona in st.session_state.PANELISTS}
    votes = {}

    # --- PHASE 1: OPENING STATEMENTS ---
    for persona, model in st.session_state.PANELISTS.items():
        historical_context = get_historical_context(persona, user_question)
        persona_desc = st.session_state.PERSONA_DESCRIPTIONS.get(persona, f"You are a {persona}.")
        prompt_template = f"""
        {persona_desc}
        {historical_context}

        You are debating the question: "{user_question}"
        
        Write a concise, single-paragraph opening statement from your perspective.
        Do not include any pre-amble.
        """
        
        statement = get_model_response(model, prompt_template)
        
        if statement:
            arguments_log[persona]["opening"] = statement
            st.markdown(f"**🎙️ {persona} ({model}):**")
            st.markdown(statement)
            transcript_log.append(f"🎙️ {persona} ({model}):\n{statement}\n" + "-"*30)
        
        time.sleep(1)

    # --- PHASE 2: REBUTTALS ---
    st.markdown("---")
    st.subheader("Phase 2: Rebuttals")
    st.write("Each panelist will now respond to the others.")
    transcript_log.append("\nPhase 2: Rebuttals\n" + "-"*30)
    
    all_args = ""
    for persona, phases in arguments_log.items():
        all_args += f"Argument from {persona}:\n{phases.get('opening', 'N/A')}\n\n"

    for persona, model in st.session_state.PANELISTS.items():
        # own_statement = arguments_log.get(persona, {}).get('opening', "my previously stated position")
        prompt_template = f"""
        {st.session_state.PERSONA_DESCRIPTIONS.get(persona)}

        The debate is on: "{user_question}"
        
        Here are all the opening statements:
        {all_args}

        Write a single, persuasive rebuttal paragraph that responds to your opponents
        from your perspective. Do not include any pre-amble.
        """
        
        rebuttal = get_model_response(model, prompt_template)
        
        if rebuttal:
            arguments_log[persona]["rebuttal"] = rebuttal
            st.markdown(f"**🗣️ {persona} ({model})'s Rebuttal:**")
            st.markdown(rebuttal)
            transcript_log.append(f"🗣️ {persona} ({model})'s Rebuttal:\n{rebuttal}\n" + "-"*30)
        time.sleep(1)

    # --- PHASE 3: THE VOTE ---
    st.markdown("---")
    st.subheader("Phase 3: The Vote")
    st.write("All arguments are in. Time for the panel to vote.")

    phase_3_header = "\nPhase 3: The Vote\n" + "-"*30
    transcript_log.append(phase_3_header)
    voting_log.append(phase_3_header)

    final_arguments_text = ""
    persona_map = {}
    i = 1
    for persona, phases in arguments_log.items():
        arg_text = phases.get('rebuttal', phases.get('opening', 'N/A'))
        final_arguments_text += f"Argument {i} ({persona}):\n{arg_text}\n\n"
        persona_map[str(i)] = persona
        i += 1
        
    for persona, model in st.session_state.PANELISTS.items():
        prompt_template = f"""
        You are an impartial judge.
        Here are the arguments:
        {final_arguments_text}
        
        Which argument was most persuasive?
        Respond *only* with the argument number (e.g., "1", "2", "3").
        """
        
        vote_raw = get_model_response(model, prompt_template)
        
        if vote_raw:
            raw_vote_line = f"{persona} ({model}) raw vote response: {vote_raw}"
            voting_log.append(raw_vote_line)
            cleaned_vote = "".join(filter(str.isdigit, vote_raw))
            if cleaned_vote and cleaned_vote[0] in persona_map:
                final_vote = cleaned_vote[0]
                votes[persona] = final_vote
                line = f"🗳️ {persona} ({model}) votes for: Argument {final_vote}"
                st.write(line)
                transcript_log.append(line)
                voting_log.append(line)
            else:
                votes[persona] = "Spoiled"
                line = f"🗳️ {persona} ({model}) spoiled its ballot."
                st.write(line)
                transcript_log.append(line)
                voting_log.append(line)
        time.sleep(1)

    # --- FINAL TALLY & LOGGING ---
    st.markdown("---")
    st.subheader("Final Results")
    transcript_log.append("\nFinal Results\n" + "-"*30)
    voting_log.append("\nFinal Results\n" + "-"*30)
    vote_counts = {"Spoiled": 0}
    for i in persona_map.keys(): vote_counts[i] = 0
    for vote in votes.values():
        if vote in vote_counts: vote_counts[vote] += 1
            
    winner_vote_num = max(vote_counts, key=vote_counts.get)
    winning_persona = "TIE"
    
    if winner_vote_num != "Spoiled" and vote_counts[winner_vote_num] > 0:
        counts = list(vote_counts.values())
        if counts.count(vote_counts[winner_vote_num]) == 1:
            winning_persona = persona_map.get(winner_vote_num, "Error")
            line = f"🎉 The winner is: {winning_persona} (Argument {winner_vote_num})! 🎉"
            st.header(line)
            transcript_log.append(line)
            voting_log.append(line)
        else:
            line = "⚖️ The vote resulted in a TIE. No winner recorded. ⚖️"
            st.header(line)
            transcript_log.append(line)
            voting_log.append(line)
    else:
        line = "⚖️ The vote was spoiled or a TIE. No winner recorded. ⚖️"
        st.header(line)
        transcript_log.append(line)
        voting_log.append(line)
        
    st.markdown("---")
    st.subheader("Final Vote Tally")
    tally_header = "\n--- Final Vote Tally ---"
    transcript_log.append(tally_header)
    voting_log.append(tally_header)
    
    for vote_num, count in vote_counts.items():
        if vote_num == "Spoiled":
            line = f"Spoiled Ballots: {count} vote(s)"
        else:
            persona = persona_map.get(vote_num, "Unknown")
            line = f"Argument {vote_num} ({persona}): {count} vote(s)"
        st.write(line)
        transcript_log.append(line)
        voting_log.append(line)

    if winning_persona != "TIE":
        log_debate_to_db(user_question, winning_persona, arguments_log)
    else:
        line = "\n[No clear winner; debate will not be logged to history.]\n"
        st.warning(line)
        transcript_log.append(line)
        voting_log.append(line)

    debate_footer = "\n==============================\n" \
                          "      Debate Concluded.             \n" \
                          "=============================="
    transcript_log.append(debate_footer)
    voting_log.append(debate_footer)
    
    save_transcript_to_file(transcript_log, user_question)
    save_vote_transcript(voting_log, user_question)

# --- NEW: Main Streamlit App Interface ---
st.title("Local AI Persona Panel 🎙️")

with st.expander("➕ Add New Panelist"):
    with st.form("new_panelist_form", clear_on_submit=True):
        new_name = st.text_input("New Panelist Name")
        new_model = st.text_input("New Panelist Ollama Model (e.g., 'phi3:medium')", value="phi3:medium")
        new_description = st.text_area("New Panelist Description", value="You are a helpful AI assistant.")
        add_button = st.form_submit_button("Add Panelist")

        if add_button and new_name:
            if new_name not in st.session_state.PANELISTS:
                st.session_state.PANELISTS[new_name] = new_model
                st.session_state.PERSONA_DESCRIPTIONS[new_name] = new_description
                st.success(f"Panelist '{new_name}' added!")
            else:
                st.warning(f"Panelist '{new_name}' already exists.")
update_config()
# --- Sidebar for viewing logs ---
st.sidebar.title("Debate Log Viewer")

# 1. Add the radio button toggle
view_choice = st.sidebar.radio(
    "Select transcript type:",
    ("Full Transcripts", "Voting Transcripts"),
    key='transcript_view' # Links to the session state we set
)

# 2. Conditionally select which file list to display
files_to_display = []
if view_choice == "Full Transcripts":
    files_to_display = st.session_state.full_log_files
    if not files_to_display:
        st.sidebar.write("No full debate transcripts found.")
else: # Must be "Voting Transcripts"
    files_to_display = st.session_state.voting_log_files
    if not files_to_display:
        st.sidebar.write("No voting transcripts found.")

# 3. Only show the selectbox if we have files
if files_to_display:
    selected_log = st.sidebar.selectbox("Select a debate to view:", files_to_display)
    
    if selected_log:
        content = None
        # Check if the selected log is already in our session_state cache
        if st.session_state.log_content_cache["file"] == selected_log:
            content = st.session_state.log_content_cache["content"]
        else:
            # If not in cache, read from disk
            try:
                with open(selected_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Save to cache
                st.session_state.log_content_cache = {"file": selected_log, "content": content}
            except Exception as e:
                content = f"Error reading file: {e}"

        # Display the content
        if "Error reading file" in content:
            st.sidebar.error(content)
        else:
            st.sidebar.text_area("Transcript", content, height=500)
# --- NEW: Main debate interface ---
st.header("Start a New Debate")
user_question_input = st.text_input("Enter the ethical question for debate:")

if st.button("Run Debate 🚀"):
    if user_question_input:
        st.info(f"Starting debate")
        run_debate(user_question_input)
    else:
        st.warning("Please enter a question for the debate.")