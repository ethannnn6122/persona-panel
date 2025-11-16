# Persona Panel

This project is a Python-based application that simulates a debate between three AI "philosophers" with distinct personas. The application takes an ethical question from the user and orchestrates a three-phase debate. The AI responses are generated using the Ollama API, allowing you to use different local language models for each persona.
The project also features a basic memory system using a ChromaDB vector store to provide historical context for arguments and an SQLite database to log past debates.

## Features

-   **Simulated Debates:** Watch AI personas debate an ethical question you provide.
-   **Customizable Personas:** Easily configure the personas and the language models that represent them.
-   **Three-Phase Debate:** The debate unfolds in three stages:
    1.  **Opening Statements:** Each persona presents its initial argument.
    2.  **Rebuttals:** Each persona counters the others' arguments.
    3.  **Voting:** Each persona votes for the most persuasive argument.
-   **Historical Context:** The application uses a ChromaDB vector store to find and provide relevant past arguments to the debaters, allowing for a semblance of memory.
-   **Debate Logging:** All debates are logged in an SQLite database, and the arguments are stored in the vector store for future reference.
-   **Transcripts:** A full transcript of each debate is saved to a text file in the `transcripts` directory.

## Requirements

-   Python 3.6+
-   [Ollama](https://ollama.ai/) installed and running.
-   Some language models you could use downloaded and available in Ollama.
-   Recommend using at lest a 7B model to ensure the model(s) stay in character.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/persona-panel.git
    cd persona-panel
    ```

2.  **Create a virtual environment and install the dependencies:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure the Debaters (two ways)**

    You can configure personas either by editing the defaults in the code (persistent) or by using the Streamlit UI (quick, in-app).

    Option A — Streamlit UI (quick, in-app)

    1.  Start the app with:

        ```bash
        streamlit run app.py
        ```

    2.  In the web UI you'll find two controls for persona management:
        - `➕ Add New Panelist` — add a new persona by providing a name, the Ollama model (e.g., `phi3:medium`) and a description, then click **Add Panelist**.
        - `⚙️ Manage Existing Panelist` — expand this to edit any existing persona. Each persona has a small form where you can rename the persona, change the Ollama model, update the description, save changes, or remove the persona entirely.

    3.  Notes about the UI method:
        - Saving changes in the UI updates Streamlit's session state and the app will rerun to show the new values.
        - Changes made via the UI apply only to the running session (they are not written to the repository files). To make permanent changes, edit the defaults in code (Option B) and restart the app.

    Option B — Edit defaults in code (persistent)

    Edit the defaults near the top of `app.py` where `st.session_state.PANELISTS` and `st.session_state.PERSONA_DESCRIPTIONS` are initialized. For example:

    ```python
    st.session_state.PANELISTS = {
        "Modern Liberal": "phi3:medium",
        "Modern Conservative": "phi3:medium",
        "Libertarian": "phi3:medium"
    }
    ```

    After saving changes to `app.py`, restart the Streamlit app to load the new defaults.

## Usage

There are two ways to run the Persona Panel application:

### Graphical User Interface

To use the web-based graphical interface, run the `app.py` script:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the application's user interface.

### Command-Line Interface

To use the command-line interface, run the `app_CLI.py` script:

```bash
python app_CLI.py
```

The application will then prompt you to enter an ethical question for the debate in your terminal.

## Application vs. Command-Line Interface

This project includes two different front-ends for the debate simulation:

-   **`app.py` (Web Application):** This is a graphical user interface built with [Streamlit](https://streamlit.io/). It allows you to run debates, view transcripts, and interact with the application through your web browser. This is the recommended way to use the application for a more user-friendly experience.

-   **`app_CLI.py` (Command-Line Interface):** This is a command-line application that runs entirely in your terminal. It provides the same core debate functionality as the web application but without a graphical interface and some additional features. This is useful for users who prefer to work in the terminal or for scripting and automation purposes.

## How It Works

1.  **Initialization:** The script initializes the SQLite and ChromaDB databases.
2.  **User Input:** The user provides an ethical question.
3.  **Opening Statements:** For each persona, the script constructs a prompt that includes the persona description, the user's question, and any relevant historical context from the vector store. It then sends this prompt to the corresponding language model via the Ollama API.
4.  **Rebuttals:** The script compiles all the opening statements and asks each persona to write a rebuttal that defends its original position and counters the other arguments.
5.  **Voting:** Each persona is asked to act as an impartial judge and vote for the most persuasive argument.
6.  **Results and Logging:** The votes are tallied, a winner is declared (if there is no tie), and the debate is logged to the databases. The winning arguments are flagged in the vector store to be used as positive examples in the future.
7.  **Transcript:** A full transcript of the debate is saved to a file.
