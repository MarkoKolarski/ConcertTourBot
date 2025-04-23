import streamlit as st
import logging
from dotenv import load_dotenv
from typing import Any
from googlesearch import search

# Load environment variables first
load_dotenv()

# --- Configure basic logging for visibility (Streamlit handles some output) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- Import your modules ---
# Add the current directory to sys.path if needed, though Streamlit usually runs from the script dir
# sys.path.append('.')

from config import (
    LLM_PROVIDER, GEMINI_API_KEY, GEMINI_API_KEY_ENV_VAR,
    CONCERT_KEYWORDS # Potentially useful for heuristic check in bonus feature
)
from repository_utils import get_repository, ConcertRAGRepository
from document_processor import is_concert_domain, summarize_document
from qa_handler import answer_question
from llm_integrator import get_llm_client # Used inside cached function
from llm_integrator import generate_qa_answer # Re-using for concert search answer generation

# --- Bonus Feature: Online Search (using the provided tool) ---
# The code interpreter environment automatically provides the Google Search tool.
# We don't need explicit SerpAPI/Bing imports here, just call the tool.

def perform_online_concert_search(artist_name: str, llm_client: Any, provider_name: str) -> str:
    """
    Performs an online search for upcoming concerts for a given artist
    and uses the LLM to synthesize an answer from the search results.

    Args:
        artist_name: The name of the musician or band.
        llm_client: The initialized LLM client object (HF dict or Gemini model).
        provider_name: The name of the active provider ('huggingface' or 'gemini').

    Returns:
        A string containing the synthesized answer or an error message.
    """
    if not artist_name:
        return "Please provide a musician or band name to search for concerts."

    logging.info(f"Performing online search for concerts by: {artist_name}")
    search_query = f"{artist_name} upcoming concerts tour dates schedule"
    search_results = ""

    try:
        # Use the available Google Search tool
        print(f"Searching online for: {search_query}") # Print to Streamlit console/logs
        search_response = [{"link": url} for url in search(search_query, num_results=10)]

        # Process search results - this is a simplified example.
        # The structure of search_response depends on the tool's output.
        # Let's assume it might have 'answerBox', 'organic_results' with 'snippet', 'title', 'link'.
        snippets = []
        if 'answerBox' in search_response[0] and 'snippet' in search_response[0]['answerBox']:
             snippets.append(search_response[0]['answerBox']['snippet'])
        if 'organic_results' in search_response[0]:
            for result in search_response[0]['organic_results']:
                if 'snippet' in result:
                    snippets.append(result['snippet'])

        if not snippets:
            logging.warning(f"No relevant search snippets found for '{artist_name}' concerts.")
            return f"Could not find upcoming concert information for {artist_name} online."

        # Join snippets to create context for the LLM
        context = "\n---\n".join(snippets)
        logging.info(f"Context from search results for LLM:\n{context[:500]}...")

        # Use the LLM to synthesize the answer from the search results
        qa_prompt = f"""Based *only* on the following online search results, summarize the upcoming concert dates, locations, or tour information for the artist "{artist_name}". If no specific dates are mentioned, state that based on the results.

Search Results:
{context}

Answer:
"""
        logging.info(f"Sending search results to LLM for synthesis ({provider_name}).")
        # Re-use the generate_qa_answer function which takes context and query
        # Here, the 'query' is implicit in the prompt, and 'context' is the search results.
        # We'll pass the prompt as query and context as empty or maybe pass the relevant snippets
        # Let's adjust generate_qa_answer slightly or create a new function if needed.
        # Looking at generate_qa_answer, it takes query and context. Let's format it like that.
        synth_answer = generate_qa_answer(f"Summarize upcoming concert dates for {artist_name} based on the search results.", context, llm_client, provider_name)


        if synth_answer.startswith("Error:"):
             logging.error(f"LLM synthesis of search results failed ({provider_name}): {synth_answer}")
             return f"Found information online, but couldn't synthesize a clear answer: {synth_answer}"

        if not synth_answer or len(synth_answer.strip()) < 10: # Basic check for empty/short answer
             return f"Found information online, but the summary is unclear. Try a different search query or check sources manually for {artist_name}."


        return f"Based on online search results:\n\n{synth_answer.strip()}"

    except NameError:
        return "Online search tool (Google Search) is not available in this environment."
    except Exception as e:
        logging.error(f"An error occurred during online concert search for '{artist_name}': {e}", exc_info=True)
        return f"An error occurred while searching for concerts for {artist_name}: {e}"


# --- Streamlit App Initialization ---

st.set_page_config(page_title="Concert Tour Info Bot", layout="wide")

st.title("Concert Tour Information Bot")
st.markdown("""
Welcome to the Concert Tour Info Bot!
You can **ADD** documents about concert tours or **QUERY** existing information.
If no documents are added, you can enter an **artist or band name** to search for their upcoming concerts online.
""")

# --- LLM Provider Selection (using session state) ---
# Initialize session state for provider if not exists
if 'llm_provider' not in st.session_state:
    # Default to Hugging Face if GEMINI_API_KEY is missing, otherwise Gemini
    st.session_state.llm_provider = 'huggingface' if not GEMINI_API_KEY else 'gemini'

st.sidebar.header("Configuration")
provider_options = {'Hugging Face (Local)': 'huggingface'}
if GEMINI_API_KEY:
     provider_options['Google Gemini (API)'] = 'gemini'

selected_provider_label = st.sidebar.radio(
    "Select LLM Provider:",
    options=list(provider_options.keys()),
    index=list(provider_options.values()).index(st.session_state.llm_provider)
)
# Update session state based on user selection
st.session_state.llm_provider = provider_options[selected_provider_label]

# Display Gemini API key status
if st.session_state.llm_provider == 'gemini':
    if GEMINI_API_KEY:
        st.sidebar.success("Gemini API Key Loaded")
    else:
         st.sidebar.error(f"Gemini API Key ({GEMINI_API_KEY_ENV_VAR}) not found in .env")
         st.warning("Gemini API key not set. Please set it in your .env file to use the Gemini provider.")


# --- Cached Resources Initialization ---
# These functions will run only once per session or when dependencies change

@st.cache_resource(hash_funcs={type(st.session_state): id})
def get_llm_client_cached(provider_name: str) -> Any:
    """Caches the initialization of the LLM client."""
    st.info(f"Initializing {provider_name.upper()} LLM client...")
    try:
        # Use the get_llm_client function from your module
        client, actual_provider_name = get_llm_client(provider_name)
        if client is None:
             st.error(f"Failed to initialize LLM client for {provider_name}.")
        else:
             st.success(f"{actual_provider_name.upper()} LLM client initialized.")
        return client
    except Exception as e:
        st.error(f"Error initializing LLM client ({provider_name}): {e}")
        logging.error(f"Error initializing LLM client ({provider_name}): {e}", exc_info=True)
        return None

@st.cache_resource
def get_repository_cached() -> ConcertRAGRepository:
    """Caches the initialization of the RAG repository."""
    st.info("Initializing RAG Repository (FAISS Index and Summary Map)...")
    try:
        # Use the get_repository function from your module
        repo = get_repository()
        st.success("RAG Repository initialized.")
        return repo
    except Exception as e:
        st.error(f"Error initializing RAG Repository: {e}")
        logging.error(f"Error initializing RAG Repository: {e}", exc_info=True)
        return None

# Initialize/Get cached resources
llm_client = get_llm_client_cached(st.session_state.llm_provider)
repository = get_repository_cached()

# Check if initialization failed
if llm_client is None or repository is None:
     st.error("Core components failed to initialize. Please check logs and configuration.")
     st.stop() # Stop the app if essential components are missing


# --- Session State for Chat History ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
st.subheader("Interaction History")
chat_history_area = st.empty() # Use a placeholder to update the history area

def display_messages():
    """Displays the messages stored in session state."""
    # Format messages for display, e.g., using Markdown
    formatted_history = ""
    for msg_type, message in st.session_state.messages:
        if msg_type == "user":
            formatted_history += f"**You:** {message}\n\n"
        elif msg_type == "bot":
            formatted_history += f"**Bot:** {message}\n\n"
        elif msg_type == "info":
             formatted_history += f"ℹ️ *{message}*\n\n"
        elif msg_type == "error":
             formatted_history += f"❌ *Error: {message}*\n\n"

    chat_history_area.markdown(formatted_history)


display_messages()


# --- User Input ---
user_input = st.text_input("Enter your request (e.g., 'ADD: <text>', 'QUERY: <question>', or Artist Name):", key="user_input")
process_button = st.button("Process Request")


# --- Process User Input ---
if process_button and user_input:
    st.session_state.messages.append(("user", user_input))
    display_messages() # Update display immediately with user message

    response = ""
    user_input_upper = user_input.upper()

    with st.spinner("Processing..."):
        try:
            if user_input_upper.startswith("ADD:"):
                document_text = user_input[4:].strip()
                if not document_text:
                    response = "Error: ADD command requires document text after 'ADD:'."
                elif is_concert_domain(document_text):
                    st.session_state.messages.append(("info", "Document seems relevant. Generating summary..."))
                    display_messages()
                    # Pass cached client and provider name
                    summary = summarize_document(document_text, llm_client, st.session_state.llm_provider)

                    if summary.startswith("Error:"):
                        response = f"Could not process the document: {summary}"
                    else:
                        doc_id = repository.add_document_summary(summary)
                        if doc_id != -1:
                            response = f"Document successfully added (ID: {doc_id}).\n\nSummary:\n'{summary}'"
                        else:
                            response = f"Error: Generated summary but failed to save it.\n\nSummary:\n'{summary}'"
                else:
                    response = "Sorry, I cannot ingest documents with other themes."

            elif user_input_upper.startswith("QUERY:"):
                query_text = user_input[6:].strip()
                if not query_text:
                    response = "Error: QUERY command requires a question after 'QUERY:'."
                else:
                    st.session_state.messages.append(("info", f"Searching and generating answer using {st.session_state.llm_provider.upper()}..."))
                    display_messages()
                    # Pass cached repository, client, and provider name
                    answer = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)
                    response = answer # answer_question already returns a formatted string including potential errors

            elif user_input_upper == "COUNT":
                 count = repository.get_total_documents()
                 response = f"Total documents currently stored: {count}"

            elif user_input_upper == "HELP":
                 response = """
Available Commands:
ADD: <document text>   - Add a new document about a concert tour.
QUERY: <your question> - Ask a question about the concert tours (RAG).
COUNT                  - Show the number of documents stored (RAG).
<Artist or Band Name>  - If no documents are loaded, search online for concerts.
"""
            # --- Bonus Feature Logic: Concert Search ---
            elif repository.get_total_documents() == 0:
                # If no documents are loaded and it's not a command, try online search
                artist_name = user_input.strip()
                if artist_name:
                    st.session_state.messages.append(("info", f"No documents loaded. Attempting online search for concerts by '{artist_name}'..."))
                    display_messages()
                    # Pass cached client and provider name
                    search_result = perform_online_concert_search(artist_name, llm_client, st.session_state.llm_provider)
                    response = search_result
                else:
                     response = "Please enter 'ADD:', 'QUERY:', or an artist name."

            # --- Default RAG Query if documents exist and not a specific command ---
            else:
                # If documents exist and it's not ADD/QUERY/COUNT/HELP, treat as implicit QUERY
                query_text = user_input.strip()
                if query_text:
                     st.session_state.messages.append(("info", f"Documents loaded. Treating input as query. Searching and generating answer using {st.session_state.llm_provider.upper()}..."))
                     display_messages()
                     # Pass cached repository, client, and provider name
                     answer = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)
                     response = answer # answer_question already returns a formatted string
                else:
                    response = "Please enter a command (ADD:, QUERY:, COUNT, HELP) or a query/artist name."


        except Exception as e:
            logging.error(f"An error occurred during request processing: {e}", exc_info=True)
            response = f"An unexpected error occurred: {e}. Please check the application logs."
            st.session_state.messages.append(("error", response)) # Add error specifically

    # Append bot response to history and update display
    st.session_state.messages.append(("bot", response))
    display_messages()


# Optional: Display repository status in sidebar
st.sidebar.markdown("---")
if repository:
    st.sidebar.info(f"Repository Status:\nDocuments: {repository.get_total_documents()}")
else:
    st.sidebar.warning("Repository failed to load.")