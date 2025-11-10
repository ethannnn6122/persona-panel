import requests
import time
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
import re
import glob  
import streamlit as st

# --- Configuration (Moved some to Streamlit UI) ---

# --- NEW: Set up the Streamlit page ---
st.set_page_config(
    page_title="Local AI Persona Panel",
    page_icon="🎙️",
    layout="wide"
)

# 1. Define your "debaters" (remains the same)
PANELISTS = {
    "Modern Liberal": "phi3:medium",
    "Modern Conservative": "phi3:medium",
    "Libertarian": "phi3:medium"
}

PERSONA_DESCRIPTIONS = {
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

# 3. The Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# 4. The Database files
DB_FILE = "debate.db"
VECTOR_DB_PATH = "vector_store"
VECTOR_COLLECTION_NAME = "debate_history"

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

# Load the models
EMBEDDING_MODEL = get_embedding_model()
CHROMA_COLLECTION = get_chroma_client()


# --- All backend functions (init_db, get_historical_context, etc.) ---
# --- MODIFIED: Replaced all 'print' with 'st.info' or 'st.warning' ---

def init_db():
    # (This function is unchanged)
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

# --- NEW: v3.1 - A "Safer" Learning Function ---
def get_historical_context(persona, question):
    """
    Queries the ChromaDB vector store for semantically similar
    past arguments. This *new* version creates a "safer" prompt
    that is less likely to confuse the models.
    """
    st.info(f"[Searching vector database for relevant history for {persona}...] \n")
    
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
    st.info("[Logging debate results to SQL and Vector databases...]")
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
                model = PANELISTS[persona]
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
    filename = f"debate_{timestamp}_{safe_question_snippet}.txt"
    
    st.info(f"Saving full transcript to: {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(transcript_log))
        st.success("[Transcript saved successfully.]")
    except IOError as e:
        st.error(f"Error saving transcript file: {e}")


# --- Main Debate Logic (Modified for Streamlit) ---

def run_debate(user_question):
    """
    Orchestrates the entire 3-phase debate and saves a transcript.
    This function now uses st.write/st.markdown to print.
    """
    
    # --- NEW: List to store the full transcript ---
    transcript_log = []
    
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
    
    arguments_log = {persona: {} for persona in PANELISTS}
    votes = {}

    # --- PHASE 1: OPENING STATEMENTS ---
    for persona, model in PANELISTS.items():
        historical_context = get_historical_context(persona, user_question)
        persona_desc = PERSONA_DESCRIPTIONS.get(persona, f"You are a {persona}.")
        prompt_template = f"""
        {persona_desc}

        Here is some context from past debates.
        Use only to inform your argument.
        {historical_context}

        **YOUR PRIMARY TASK:**
        Write a concise, one-paragraph opening statement that *only*
        addresses the new question: "{user_question}"
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

    for persona, model in PANELISTS.items():
        # own_statement = arguments_log.get(persona, {}).get('opening', "my previously stated position")
        prompt_template = f"""
        {PERSONA_DESCRIPTIONS.get(persona)}

        Here are the opening statements from your opponents:
        {all_args}

        **Your Task:**
        Write a single, persuasive paragraph that *rebuts* your opponents' arguments from *your* perspective.
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
    transcript_log.append("\nPhase 3: The Vote\n" + "-"*30)

    final_arguments_text = ""
    persona_map = {}
    i = 1
    for persona, phases in arguments_log.items():
        arg_text = phases.get('rebuttal', phases.get('opening', 'N/A'))
        final_arguments_text += f"Argument {i} ({persona}):\n{arg_text}\n\n"
        persona_map[str(i)] = persona
        i += 1
        
    for persona, model in PANELISTS.items():
        prompt_template = f"""
        You are now an impartial judge...
        Which argument do you vote for? 
        Respond *only* with the argument number (e.g., "1", "2", or "3").
        """
        
        vote_raw = get_model_response(model, prompt_template)
        
        if vote_raw:
            cleaned_vote = "".join(filter(str.isdigit, vote_raw))
            if cleaned_vote and cleaned_vote[0] in persona_map:
                final_vote = cleaned_vote[0]
                votes[persona] = final_vote
                line = f"🗳️ {persona} ({model}) votes for: Argument {final_vote}"
                st.write(line)
                transcript_log.append(line)
            else:
                votes[persona] = "Spoiled"
                line = f"🗳️ {persona} ({model}) spoiled its ballot."
                st.write(line)
                transcript_log.append(line)
        time.sleep(1)

    # --- FINAL TALLY & LOGGING ---
    st.markdown("---")
    st.subheader("Final Results")
    transcript_log.append("\nFinal Results\n" + "-"*30)
    
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
        else:
            line = "⚖️ The vote resulted in a TIE. No winner recorded. ⚖️"
            st.header(line)
            transcript_log.append(line)
    else:
        line = "⚖️ The vote was spoiled or a TIE. No winner recorded. ⚖️"
        st.header(line)
        transcript_log.append(line)
        
    st.markdown("---")
    st.subheader("Final Vote Tally")
    transcript_log.append("\n--- Final Vote Tally ---")
    
    for vote_num, count in vote_counts.items():
        if vote_num == "Spoiled":
            line = f"Spoiled Ballots: {count} vote(s)"
        else:
            persona = persona_map.get(vote_num, "Unknown")
            line = f"Argument {vote_num} ({persona}): {count} vote(s)"
        st.write(line)
        transcript_log.append(line)

    if winning_persona != "TIE":
        log_debate_to_db(user_question, winning_persona, arguments_log)

    else:
        line = "\n[No clear winner; debate will not be logged to history.]\n"
        st.warning(line)
        transcript_log.append(line)

    transcript_log.append("\n==============================\n" \
                          "      Debate Concluded.             \n" \
                          "==============================")
    
    save_transcript_to_file(transcript_log, user_question)


# --- NEW: Main Streamlit App Interface ---

st.title("Local AI Persona Panel 🎙️")

# --- NEW: Sidebar for viewing logs ---
st.sidebar.title("Debate Log Viewer")
log_files = sorted(glob.glob("debate_*.txt"), reverse=True)
if not log_files:
    st.sidebar.write("No debate transcripts found.")
else:
    selected_log = st.sidebar.selectbox("Select a debate to view:", log_files)
    if selected_log:
        try:
            with open(selected_log, 'r', encoding='utf-8') as f:
                st.sidebar.text_area("Transcript", f.read(), height=500)
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# --- NEW: Main debate interface ---
st.header("Start a New Debate")
user_question_input = st.text_input("Enter the ethical question for debate:")

if st.button("Run Debate 🚀"):
    if user_question_input:
        # This is where the magic happens
        # We call the main function, and all st.write()
        # calls inside it will stream to the UI.
        run_debate(user_question_input)
    else:
        st.warning("Please enter a question for the debate.")

# --- NEW: Run the DB init on startup ---
if __name__ == "__main__":
    init_db()