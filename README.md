# Persona Panel

This project is a Python-based application that simulates a debate between three AI "philosophers" with distinct personas: a Modern Liberal, a Modern Conservative, and a Libertarian. The application takes an ethical question from the user and orchestrates a three-phase debate. The AI responses are generated using the Ollama API, allowing you to use different local language models for each persona.

The project also features a basic memory system using a ChromaDB vector store to provide historical context for arguments and an SQLite database to log past debates.

## Features

-   **Simulated Debates:** Watch three AI personas debate an ethical question you provide.
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
-   The language models you want to use (e.g., `phi3:mini`, `gemma:2b`, `qwen:4b`) downloaded and available in Ollama.

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
    *(Note: A `requirements.txt` file does not currently exist. You will need to create one based on the imports in `app.py`)*

3.  **Configure the Debators:**
    Open `app.py` and edit the `PANELISTS` dictionary to match the models you have downloaded in Ollama. You can also edit the `PERSONA_DESCRIPTIONS` to change the behavior of the debaters.

    ```python
    PANELISTS = {
        "Modern Liberal": "phi3:mini",
        "Modern Conservative": "gemma:2b",
        "Libertarian": "qwen:4b"
    }
    ```

## Usage

To run the application, simply execute the `app.py` script:

```bash
python app.py
```

The application will then prompt you to enter an ethical question for the debate. The debate will proceed, and a transcript will be saved in the `transcripts` directory.

## How It Works

1.  **Initialization:** The script initializes the SQLite and ChromaDB databases.
2.  **User Input:** The user provides an ethical question.
3.  **Opening Statements:** For each persona, the script constructs a prompt that includes the persona description, the user's question, and any relevant historical context from the vector store. It then sends this prompt to the corresponding language model via the Ollama API.
4.  **Rebuttals:** The script compiles all the opening statements and asks each persona to write a rebuttal that defends its original position and counters the other arguments.
5.  **Voting:** Each persona is asked to act as an impartial judge and vote for the most persuasive argument.
6.  **Results and Logging:** The votes are tallied, a winner is declared (if there is no tie), and the debate is logged to the databases. The winning arguments are flagged in the vector store to be used as positive examples in the future.
7.  **Transcript:** A full transcript of the debate is saved to a file.
