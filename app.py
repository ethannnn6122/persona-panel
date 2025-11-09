import requests
import time
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import datetime
import pathlib

# Program Goal:
# Define the 3 "philosopher" personas.
# Define the downloaded models.
# Take your ethical question.
# Execute the 3-phase debate (Opening, Rebuttal, Vote).
# Print a full transcript of the debate.

# --- Configuration ---

# 1. Define your "debaters" by mapping a persona to a model
# Make sure these model names match your 'ollama list'
PANELISTS = {
    "Modern Liberal": "phi3:mini",
    "Modern Conservative": "gemma:2b",
    "Libertarian": "qwen:4b"
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

# 2. The question
USER_QUESTION = input("Enter the ethical question for debate: ")

# 3. The Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# # 4. The Database file 
DB_FILE = "debate.db"
VECTOR_DB_PATH = "vector_store"
VECTOR_COLLECTION_NAME = "debate_history"

# Init Embedding Model and Vector DB
print("\n[Initializing vector database for historical context...]\n")
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("[Embedding model loaded successfully.]\n")
except Exception as e:
    print(f"Error loading embedding model: {e}\n")
    print("Please ensure 'sentence-transformers' is installed and the model is available.\n")
    exit()
try:
    CHROMA_CLIENT = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(name=VECTOR_COLLECTION_NAME)
    print(f"[Vector database connected at '{VECTOR_DB_PATH}'.]\n")
except Exception as e:
    print(f"Error initializing vector database: {e}\n")
    print("Please ensure 'chromadb' is installed and the path is accessible.\n")
    exit()

# --- End of Configuration ---

def init_db():
    """
    Initializes the SQLite database to store debate transcripts.
    and the ChromaDB collection for vector embeddings.
    """
    # SQL DB
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS debates (
        debate_id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        winning_persona TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create 'arguments' table
    # This stores every single argument made
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS arguments (
        argument_id INTEGER PRIMARY KEY AUTOINCREMENT,
        debate_id INTEGER NOT NULL,
        persona TEXT NOT NULL,
        model TEXT NOT NULL,
        phase TEXT NOT NULL,  -- e.g., "opening", "rebuttal"
        argument_text TEXT NOT NULL,
        FOREIGN KEY (debate_id) REFERENCES debates (debate_id)
        )
    ''')

    conn.commit()
    conn.close()

    # ChromaDB collection is initialized gloabally
    print("SQL database initialized.\n")

def get_historical_context(persona, question):
    """
    Queries the ChromaDB vector store for semantically similar
    past arguments to build a relevant history.
    """
    print(f"\n[Searching vector database for relevant history for {persona}...] \n")
    try:
        question_embedding = EMBEDDING_MODEL.encode(question).tolist()
        results = CHROMA_COLLECTION.query(
            query_embeddings=[question_embedding],
            n_results=2,
            where={"persona": persona} # Filter by persona
        )

        history = ""
        if results["documents"] and results["metadatas"]:
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                outcome = "WON" if meta.get('is_winner') else "LOST"
                history += f"**A similar argument you made in the past that {outcome}:**\n\"{doc}\"\n\n"
        if history:
            print(history)
            return "--- RELEVANT HISTORICAL CONTEXT ---\n" + history + "--- END OF CONTEXT ---\n"
        else:
            return "No relevant history found for this specifc topic. Good luck!\n"
    except Exception as e:
        print(f"Error querying vector database: {e}\n")
        return "No relevant history found due to an error. Proceeding without context.\n"
    
def log_debate_to_db(question, winning_persona, arguments_log):
    """Logs the entire debate and its outcome to the databases."""
    print("\n[Logging debate results to databases...]\n")
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 1. Log the debate
        cursor.execute("INSERT INTO debates (question, winning_persona) VALUES (?, ?)", (question, winning_persona))
        debate_id = cursor.lastrowid # Get the ID of the debate we just logged
        
        chroma_ids = []
        chroma_documents = []
        chroma_metadatas = []

        # 2. Log all arguments
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
        print("Debate successfully logged to SQL DB.")

        if chroma_documents:
            embeddings = EMBEDDING_MODEL.encode(chroma_documents).tolist()

            CHROMA_COLLECTION.add(
                embeddings=embeddings,
                documents=chroma_documents,
                metadatas=chroma_metadatas,
                ids=chroma_ids
            )
            print("Debate arguments successfully logged to vector database.\n")
        
    except Exception as e:
        print(f"Error logging to database: {e}")

def get_model_response(model_name, prompt):
    """
    Sends a prompt to a specific model running in Ollama and gets a response.
    """
    print(f"[Querying {model_name} for its thoughts...]\n")
    
    # This is the data we send to the Ollama API
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # We want the full response at once
    }
    
    try:
        # Make the API call
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # 5 min timeout
        response.raise_for_status()  # Raise an error for bad responses
        
        # Parse the JSON response
        data = response.json()
        
        # The 'response' field contains the model's full text
        return data.get("response").strip()
        
    except requests.exceptions.Timeout:
        print(f"Error: The request to {model_name} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama API for {model_name}: {e}")
        return None

def save_transcript_to_file(transcript_log):
    """Saves the collected transcript log to a timestamped text file."""
    
    # Create a unique, clean filename
    transcript_dir = pathlib.Path("transcripts")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"debate_{timestamp}.txt"
    
    print(f"\n[Saving full transcript to: {filename}]")

    try:
        transcript_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
        file_path = transcript_dir / filename
        file_path.write_text("\n".join(transcript_log), encoding="utf-8")
        print("[Transcript saved successfully.]")
    except IOError as e:
        print(f"Error saving transcript file: {e}")

def run_debate():
    """
    Orchestrates the entire 3-phase debate and saves a transcript.
    """
    
    # --- NEW: List to store the full transcript ---
    transcript_log = []
    
    header = "=============================================="
    title = "    Ethical Debate Panel Has Convened!     "
    
    print(header)
    print(title)
    print(header)
    transcript_log.append(header + "\n" + title + "\n" + header)
    
    question_line = f"\nTonight's question: {USER_QUESTION}\n"
    print(question_line)
    transcript_log.append(question_line)

    phase1_header = "Let's begin with opening statements...\n"
    divider = "----------------------------------------------\n"
    print(phase1_header)
    print(divider)
    transcript_log.append(phase1_header + "\n" + divider)
    
    arguments_log = {persona: {} for persona in PANELISTS}
    votes = {}

    # --- PHASE 1: OPENING STATEMENTS ---
    for persona, model in PANELISTS.items():
        historical_context = get_historical_context(persona, USER_QUESTION)
        
        persona_desc = PERSONA_DESCRIPTIONS.get(persona, f"You are a {persona}.")

        prompt_template = f"""
        {persona_desc}
        Your goal is to form the most persuasive and ethically 
        sound argument to win this debate.
        {historical_context}
        **Your Task:**
        Analyze this ethical question: "{USER_QUESTION}"
        Write a concise, one-paragraph opening statement.
        Learn from history, but stay true to your core principles.
        """
        
        statement = get_model_response(model, prompt_template)
        
        if statement:
            arguments_log[persona]["opening"] = statement
            
            # --- NEW: Log to transcript ---
            line = f" {persona} ({model}):\n{statement}\n"
            print(line)
            print(divider)
            transcript_log.append(line + "\n" + divider)
        
        time.sleep(1)

    # --- PHASE 2: REBUTTALS ---
    phase2_header = "\n==============================================\n" \
                    "           Phase 2: Rebuttals            \n" \
                    "==============================================\n" \
                    "\nEach panelist will now respond to the others.\n"
    print(phase2_header)
    transcript_log.append(phase2_header)
    
    all_args = ""
    for persona, phases in arguments_log.items():
        all_args += f"Argument from {persona}:\n{phases.get('opening', 'N/A')}\n\n"

    for persona, model in PANELISTS.items():
        own_statement = arguments_log.get(persona, {}).get('opening', "my previously stated position")
        
        prompt_template = f"""
        You are a {persona} philosopher.
        Your initial argument was: "{own_statement}"
        Now, consider the *other* arguments on the table:
        {all_args}
        Write a one-paragraph rebuttal that defends your 
        original position and *counters the other arguments*. 
        Your goal is to win the vote.
        """
        
        rebuttal = get_model_response(model, prompt_template)
        
        if rebuttal:
            arguments_log[persona]["rebuttal"] = rebuttal
            
            # --- NEW: Log to transcript ---
            line = f" {persona} ({model})'s Rebuttal:\n{rebuttal}\n"
            print(line)
            print(divider)
            transcript_log.append(line + "\n" + divider)
            
        time.sleep(1)

    # --- PHASE 3: THE VOTE ---
    phase3_header = "\n==============================================\n" \
                    "            Phase 3: The Vote              \n" \
                    "==============================================\n" \
                    "\nAll arguments are in. Time for the panel to vote.\n"
    print(phase3_header)
    transcript_log.append(phase3_header)

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
        You are now an impartial judge. Your personal philosophy is irrelevant.
        Your task is to vote on which of the following arguments is the 
        most persuasive, well-reasoned, and ethically sound solution.
        Here are the final arguments:
        {final_arguments_text}
        Which argument do you vote for? 
        Respond *only* with the argument number (e.g., "1", "2", or "3").
        """
        
        vote_raw = get_model_response(model, prompt_template)
        
        if vote_raw:
            cleaned_vote = "".join(filter(str.isdigit, vote_raw))
            if cleaned_vote and cleaned_vote[0] in persona_map:
                final_vote = cleaned_vote[0]
                votes[persona] = final_vote
                
                # --- NEW: Log to transcript ---
                line = f"{persona} ({model}) votes for: Argument {final_vote}\n"
                print(line)
                transcript_log.append(line)
            else:
                votes[persona] = "Spoiled"
                
                # --- NEW: Log to transcript ---
                line = f" {persona} ({model}) spoiled its ballot.\n"
                print(line)
                transcript_log.append(line)
        
        time.sleep(1)

    # --- FINAL TALLY & LOGGING ---
    results_header = "\n==============================================\n" \
                     "            Final Results              \n" \
                     "==============================================\n"
    print(results_header)
    transcript_log.append(results_header)
    
    vote_counts = {"Spoiled": 0}
    for i in persona_map.keys():
        vote_counts[i] = 0

    for vote in votes.values():
        if vote in vote_counts:
            vote_counts[vote] += 1
            
    winner_vote_num = max(vote_counts, key=vote_counts.get)
    winning_persona = "TIE"
    
    if winner_vote_num != "Spoiled" and vote_counts[winner_vote_num] > 0:
        counts = list(vote_counts.values())
        if counts.count(vote_counts[winner_vote_num]) == 1:
            winning_persona = persona_map.get(winner_vote_num, "Error")
            
            # --- NEW: Log to transcript ---
            line = f"\n The winner is: {winning_persona} (Argument {winner_vote_num})!\n"
            print(line)
            transcript_log.append(line)
        else:
            line = "\n The vote resulted in a TIE. No winner recorded.\n"
            print(line)
            transcript_log.append(line)
    else:
        line = "\nThe vote was spoiled or a TIE. No winner recorded.\n"
        print(line)
        transcript_log.append(line)
        
    tally_header = "--- Final Vote Tally ---"
    print(tally_header)
    transcript_log.append("\n" + tally_header)
    
    for vote_num, count in vote_counts.items():
        if vote_num == "Spoiled":
            line = f"Spoiled Ballots: {count} vote(s)"
        else:
            persona = persona_map.get(vote_num, "Unknown")
            line = f"Argument {vote_num} ({persona}): {count} vote(s)"
        print(line)
        transcript_log.append(line)

    if winning_persona != "TIE":
        log_debate_to_db(USER_QUESTION, winning_persona, arguments_log)
    else:
        line = "\n[No clear winner; debate will not be logged to history.]\n"
        print(line)
        transcript_log.append(line)

    end_line = "==============================================" \
               "\n          Debate Concluded.             \n" \
               "=============================================="
    print(end_line)
    transcript_log.append("\n" + end_line)
    
    # --- NEW: Call the function to save the file ---
    save_transcript_to_file(transcript_log)


# --- This line runs the whole program ---
if __name__ == "__main__":
    init_db()
    run_debate()