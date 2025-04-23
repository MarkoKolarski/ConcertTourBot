import streamlit as st
import logging
from dotenv import load_dotenv
from typing import Any
import requests # Used for SerpAPI
import os

# Load environment variables first
load_dotenv()

# --- Configure basic logging for visibility (Streamlit handles some output) ---
# Log messages will appear in the terminal where you run streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- Import your modules ---
# Ensure these modules are accessible in your environment/PYTHONPATH
from repository_utils import get_repository, ConcertRAGRepository
from document_processor import is_concert_domain, summarize_document
from qa_handler import answer_question
from llm_integrator import get_llm_client, generate_qa_answer

from config import (
    GEMINI_API_KEY, GEMINI_API_KEY_ENV_VAR
    # Assuming CONCERT_KEYWORDS are used internally by your modules and don't need explicit import here for the UI file
)

# --- Helper function for online search (Kept as is - uses SerpAPI) ---
def perform_online_concert_search(artist_name: str, llm_client: Any, provider_name: str) -> str:
    """
    Performs an online search for upcoming concerts for a given artist
    and uses the LLM to synthesize an answer from the search results via SerpAPI.
    """
    if not artist_name:
        return "Please provide a musician or band name to search for concerts."

    serpapi_key = os.getenv("SERPAPI_KEY")
    # Although the key is checked before calling this function,
    # perform_online_concert_search still needs access to it.
    if not serpapi_key:
         # This case should ideally be caught before calling, but adding a fallback return
         # just in case. The calling logic provides the primary user error message.
         logging.error("perform_online_concert_search called without SerpAPI key present.")
         return "Error: SerpAPI key is missing, cannot perform online search."


    logging.info(f"Performing online search for concerts by: {artist_name}")
    search_query = f"{artist_name} upcoming concerts tour dates 2025"

    try:
        # SerpAPI search
        url = "https://serpapi.com/search"
        params = {
            "q": search_query,
            "api_key": serpapi_key, # Use the variable directly
            "engine": "google"
        }

        logging.info(f"Sending request to SerpAPI for: {search_query}")
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        search_data = response.json()

        if "error" in search_data:
            logging.error(f"SerpAPI error: {search_data['error']}")
            return f"Error performing search: {search_data['error']}"

        search_results = []

        # Check for organic results
        if "organic_results" in search_data:
            for result in search_data.get("organic_results", [])[:5]: # Limit to first 5 results, use .get for safety
                if "snippet" in result:
                    search_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    })

        # Check for answer box if available
        if "answer_box" in search_data and "snippet" in search_data["answer_box"]:
             # Use .get for safety, check if snippet is not empty
             if search_data["answer_box"].get("snippet"):
                 search_results.insert(0, {
                     "title": search_data["answer_box"].get("title", "Featured Snippet"),
                     "snippet": search_data["answer_box"].get("snippet", ""),
                     "link": search_data["answer_box"].get("link", "")
                 })

        # Check for knowledge graph if available
        if "knowledge_graph" in search_data and "description" in search_data["knowledge_graph"]:
            # Use .get for safety, check if description is not empty
            if search_data["knowledge_graph"].get("description"):
                search_results.insert(0, {
                    "title": search_data["knowledge_graph"].get("title", artist_name),
                    "snippet": search_data["knowledge_graph"].get("description", ""),
                    "link": search_data["knowledge_graph"].get("link", "") # Knowledge graph can have a link too
                })

        if not search_results:
            logging.warning(f"No search results found for {artist_name} concerts")
            return f"No concert information found for {artist_name}. They may not have announced any upcoming concerts yet or the search did not return relevant results."

        # Prepare context for LLM synthesis
        context = f"Search results for '{artist_name} upcoming concerts':\n\n"
        for i, result in enumerate(search_results):
            # Format results clearly for the LLM
            context += f"--- Result {i+1} ---\n"
            if result.get("title"):
                context += f"Title: {result['title']}\n"
            if result.get("snippet"):
                 context += f"Snippet: {result['snippet']}\n"
            if result.get("link"):
                 context += f"Link: {result['link']}\n"
            context += "\n" # Add space between results

        # Ensure context isn't excessively long
        max_context_length = 3500 # Adjust based on LLM capabilities and prompt length
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... (context truncated)"
            logging.warning(f"Search context truncated to {max_context_length} characters.")


        # Prompt for LLM synthesis
        llm_query = f"Summarize upcoming concert details for {artist_name} based *only* on the provided search results."

        logging.info(f"Sending search results to {provider_name} for synthesis")
        concert_info = generate_qa_answer(llm_query, context, llm_client, provider_name)


        if concert_info.startswith("Error:"):
            logging.error(f"LLM synthesis error: {concert_info}")
            return f"Found information online, but couldn't synthesize a clear answer: {concert_info}"

        # Check if the synthesized answer is generic or indicates no info
        lowercased_synth_answer = concert_info.strip().lower()
        if not lowercased_synth_answer or \
           len(lowercased_synth_answer) < 30 or \
           "could not find" in lowercased_synth_answer or \
           "based on the available information" in lowercased_synth_answer or \
           "i couldn't generate a specific answer" in lowercased_synth_answer or \
           "no specific dates" in lowercased_synth_answer or \
           "no concert information was found based on these results" in lowercased_synth_answer:
             return f"Found online information for {artist_name}, but the bot could not synthesize specific upcoming concert dates from the search results."


        return f"Information about {artist_name}'s upcoming concerts:\n\n{concert_info.strip()}"

    except requests.exceptions.RequestException as req_err:
        logging.error(f"SerpAPI request error: {req_err}", exc_info=True)
        return f"Network or request error during search for {artist_name}: {str(req_err)}. Check your SerpAPI key and network connection."
    except Exception as e:
        logging.error(f"An unexpected error occurred in concert search: {str(e)}", exc_info=True)
        return f"An unexpected error occurred while searching for concerts for {artist_name}: {str(e)}. Please check the logs for details."


# --- Streamlit App Initialization ---

# --- Page Configuration ---
st.set_page_config(
    page_title="Concert Bot 🎶", # Enhanced title
    layout="wide", # Already wide
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# --- Custom Styling (Simple Markdown based) ---
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
/* Adjust Streamlit's native chat bubbles (requires Streamlit 1.22+) */
.stChatMessage {
    /* Streamlit applies default styling */
}
.stChatMessage[data-testid="stChatMessage"]:has(.stMarkdown) {
    /* Optional: Add custom padding/margins if default is not enough */
    margin-bottom: 10px;
}

/* Adjust padding/margin for the main content area */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 5%; /* Add horizontal padding */
    padding-right: 5%; /* Add horizontal padding */
}
/* Center the title slightly better (optional, depends on Streamlit version) */
h1 {
    text-align: center;
    color: #004d40; /* Dark cyan */
    margin-bottom: 0.5em; /* Add some space below title */
}
/* Style for info/success/error messages */
.stAlert {
    border-radius: 10px;
    margin-bottom: 10px; /* Add space below alerts */
}
/* Adjust sidebar width */
section[data-testid="stSidebar"] {
    width: 300px !important;
    padding-right: 1rem; /* Add some space inside sidebar */
}
/* Style for the radio buttons if needed */
.stRadio > label {
    /* Add custom styles for radio button labels if desired */
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True) # Allow HTML for basic styling

# --- Title and Description ---
st.title("Concert Bot 🎶")

# Use a container for the welcome message and instructions
with st.container(border=True): # Added border for visual separation
    st.markdown("""
    Welcome to the **Concert Tour Information Bot**! I can help you manage information about
    upcoming concert tours and find details about artists' live performances.
    """)
    st.markdown("""
    Type your request below. You can use commands or just enter a query/artist name:
    - **ADD:** `<document text>`: Add a new document about a concert tour.
    - **QUERY:** `<your question>`: Ask a question based on added documents.
    - **COUNT**: See how many documents are stored.
    - Enter **Artist or Band Name**: Search online for concerts if RAG finds no info or repo is empty.
    """)

    # --- New section explaining API keys ---
    st.markdown("---") # Separator before API key explanation
    st.markdown("#### 🔑 API Key Requirements")
    st.markdown(f"""
    This application uses external services that require API keys for full functionality:
    - **`{GEMINI_API_KEY_ENV_VAR}`**: Required for the **Google Gemini** LLM provider. If this key is missing, you will only be able to use the Hugging Face (Local) provider.
    - **`SERPAPI_KEY`**: Required for the **Online Concert Search** functionality. If this key is missing, the bot cannot perform web searches for concert dates when needed.

    Please add these keys to your `.env` file located in the project's root directory.
    """)
    # --- End new section ---


# --- Sidebar Configuration ---
st.sidebar.header("Settings")

# LLM Provider Selection
# Set the initial provider to 'gemini' if it hasn't been set yet
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = 'gemini' # Set Gemini as default

# Define all provider options, regardless of key presence
provider_options = {
    'Google Gemini (API)': 'gemini',
    'Hugging Face (Local)': 'huggingface'
}

# Find the index of the currently selected provider value ('gemini' or 'huggingface')
# in the list of *values* to set the initial position of the radio button.
try:
    default_index = list(provider_options.values()).index(st.session_state.llm_provider)
except ValueError:
    # Fallback to the first option if the session state somehow holds an invalid value
    default_index = 0


# Display the radio button with ALL options
selected_provider_label = st.sidebar.radio(
    "Select LLM Provider:",
    options=list(provider_options.keys()), # Use all keys for options
    index=default_index, # Use the calculated index
    key="provider_radio"
)

# Update session state based on user selection if it changed
selected_provider_value = provider_options[selected_provider_label]
if st.session_state.llm_provider != selected_provider_value:
     st.session_state.llm_provider = selected_provider_value
     # Streamlit handles re-running and re-caching when this session state changes


# --- Display API Key Status ---
st.sidebar.markdown("---") # Separator
st.sidebar.subheader("API Status")

# Status for Gemini API Key
if GEMINI_API_KEY:
    st.sidebar.success("✅ Gemini API Key Loaded")
else:
    st.sidebar.warning(f"⚠️ Gemini API Key ({GEMINI_API_KEY_ENV_VAR}) not found")

# Status for SerpAPI Key (for online search)
serpapi_key = os.getenv("SERPAPI_KEY")
if serpapi_key:
    st.sidebar.success("✅ SerpAPI Key Loaded (Online Search Enabled)")
else:
    st.sidebar.warning("⚠️ SerpAPI Key not found")


@st.cache_resource(hash_funcs={str: id}) # Hash string argument by value
def get_llm_client_cached(provider_name: str) -> Any:
    """Caches the initialization of the LLM client."""
    # Check for Gemini API key *before* attempting initialization if Gemini is selected
    if provider_name == 'gemini' and not GEMINI_API_KEY:
        # Don't attempt initialization if key is missing, return None immediately
        logging.error(f"Attempted to initialize Gemini LLM, but GEMINI_API_KEY is missing.")
        st.error(f"Cannot initialize Google Gemini LLM. {GEMINI_API_KEY_ENV_VAR} is not set.")
        return None
        
    st.info(f"Initializing {provider_name.upper()} LLM client...")
    try:
        client, actual_provider_name = get_llm_client(provider_name)
        if client is None:
            # Error message already logged/displayed by get_llm_client if it failed for other reasons
            st.error(f"Failed to initialize LLM client for {provider_name}. Check logs.")
            return None
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
        repo = get_repository()
        st.success("RAG Repository initialized.")
        return repo
    except Exception as e:
        st.error(f"Error initializing RAG Repository: {e}")
        logging.error(f"Error initializing RAG Repository: {e}", exc_info=True)
        return None


# --- Cached Resources Initialization (with visual feedback) ---
# Wrap initialization in st.status for better visual feedback during startup
# Streamlit automatically handles the 'Initializing...' message for @st.cache_resource
# when used with st.status context manager.

# Initialize LLM Client inside a status block
# Use the currently selected provider from session state
with st.status(f"Initializing {st.session_state.llm_provider.upper()} LLM...", expanded=True) as status_llm:
    # get_llm_client_cached handles the API key check and returns None if missing
    llm_client = get_llm_client_cached(st.session_state.llm_provider)

    if llm_client:
        status_llm.update(label=f"{st.session_state.llm_provider.upper()} LLM initialized.", state="complete", expanded=False)
    else:
        status_llm.update(label=f"Failed to initialize {st.session_state.llm_provider.upper()} LLM.", state="error", expanded=True)
        # Additional fatal error message might be redundant if get_llm_client_cached already displayed one,
        # but ensures user sees something clear.

        st.stop() # Stop execution if LLM is not available


# Initialize Repository inside a status block
with st.status("Initializing RAG Repository...", expanded=True) as status_repo:
    repository = get_repository_cached()
    if repository:
        status_repo.update(label=f"RAG Repository initialized. Documents loaded: {repository.get_total_documents()}", state="complete", expanded=False)
    else:
        status_repo.update(label="Failed to initialize RAG Repository.", state="error", expanded=True)
        st.error("Fatal Error: Could not initialize the RAG Repository.")
        # Stopping here as repo is essential.
        st.stop()

# --- Session State for Chat History ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Interaction History Display ---
st.subheader("Interaction History")

# Use a container to hold the chat messages with a potential fixed height and scroll
chat_container = st.container(height=400) # Use a fixed height container


def display_messages():
    """Displays the messages stored in session state within the chat container."""
    # Clear previous content in the container before redrawing
    chat_container.empty()

    with chat_container:
        for msg_type, message in st.session_state.messages:
            # Use st.chat_message for modern chat bubble look (requires Streamlit 1.22+)
            # Role 'user' and 'assistant' trigger default bubble styling
            if msg_type == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            elif msg_type == "bot":
                 with st.chat_message("assistant"): # 'assistant' is the role for the bot
                    st.markdown(message)
            # Display info/error messages outside chat bubbles but still in the history area
            elif msg_type == "info":
                 st.info(f"ℹ️ {message}")
            elif msg_type == "error":
                 st.error(f"❌ {message}")


# Display messages initially and whenever state updates
display_messages()


# --- User Input Area ---
# Use a container for the input and button, placed below the chat history
input_container = st.container()

with input_container:
    user_input = st.text_input(
        "Enter your request:",
        placeholder="e.g., 'ADD: document text', 'QUERY: question', or 'Billie Eilish'",
        key="user_input_widget",
        label_visibility="collapsed" # Hide the default label
    )

    # Center the button (using columns)
    col1, col2, col3 = st.columns([1, 1, 1]) # [left_space, button_column, right_space]
    with col2: # Place button in the middle column
        process_button = st.button("Send Request", use_container_width=True)


# --- Process User Input ---
# The logic inside this block remains the same, only presentation changes
if process_button and user_input:
    # Append user message immediately to show it's received
    st.session_state.messages.append(("user", user_input))
    # Redraw messages with the new user input added
    display_messages()

    response = ""
    user_input_upper = user_input.upper()
    query_text = user_input.strip()

    # Use st.status for live feedback during processing
    with st.status("Processing request...", expanded=True) as processing_status:
        try:
            # 1. Handle Explicit Commands
            if user_input_upper.startswith("ADD:"):
                document_text = user_input[4:].strip()
                if not document_text:
                    response = "Error: ADD command requires document text after 'ADD:'."
                    processing_status.update(label="ADD command invalid.", state="error")
                elif is_concert_domain(document_text):
                    processing_status.update(label="Document seems relevant. Generating summary...", state="running")
                    summary = summarize_document(document_text, llm_client, st.session_state.llm_provider)
                    if summary.startswith("Error:"):
                        response = f"Could not process the document: {summary}"
                        processing_status.update(label="Summary generation failed.", state="error")
                    else:
                        processing_status.update(label="Summary generated. Ingesting into RAG system...", state="running")
                        doc_id = repository.add_document_summary(summary)
                        if doc_id != -1:
                             response = f"Document successfully added (ID: {doc_id}).\n\nSummary:\n'{summary}'"
                             processing_status.update(label="Document successfully added.", state="complete")
                        else:
                             response = f"Error: Generated summary but failed to save it.\n\nSummary:\n'{summary}'"
                             processing_status.update(label="Document saving failed.", state="error")
                else:
                    response = "Sorry, I cannot ingest documents with other themes."
                    processing_status.update(label="Document not relevant.", state="complete")

            elif user_input_upper.startswith("QUERY:"):
                 query_text_from_command = user_input[6:].strip()
                 if not query_text_from_command:
                     response = "Error: QUERY command requires a question after 'QUERY:'."
                     processing_status.update(label="QUERY command invalid.", state="error")
                 else:
                    processing_status.update(label=f"Searching RAG for '{query_text_from_command}' using {st.session_state.llm_provider.upper()}...", state="running")
                    response = answer_question(query_text_from_command, repository, llm_client, st.session_state.llm_provider)
                    processing_status.update(label="RAG query processed.", state="complete")

            elif user_input_upper == "COUNT":
                 count = repository.get_total_documents()
                 response = f"Total documents currently stored: {count}"
                 processing_status.update(label="Document count retrieved.", state="complete")

            elif user_input_upper == "HELP":
                 response = """
Available Commands:
- **ADD:** `<document text>`: Add a new document about a concert tour.
- **QUERY:** `<your question>`: Ask a question based on added documents.
- **COUNT**: See how many documents are stored.
- Enter **Artist or Band Name**: Search online for concerts if RAG finds no info or repo is empty.
""" # Improved formatting for HELP
                 processing_status.update(label="Help information displayed.", state="complete")

            # 2. Handle Non-Command Input (Potential RAG Query with Online Search Fallback)
            else:
                 current_doc_count = repository.get_total_documents()

                 # Define phrases that indicate RAG found no info (case-insensitive)
                 not_found_phrases_rag = [
                     "couldn't find specific information related to your query in the ingested documents",
                     "i couldn't find specific information",
                     "based on the available information, i couldn't generate",
                     "no information about",
                     "no relevant summaries found",
                     "could not find specific upcoming concert information",
                      "no concert information was found based on the search results"
                 ]

                 serpapi_key_present = os.getenv("SERPAPI_KEY")

                 # If repository is empty OR RAG found nothing, attempt online search (if key is present)
                 attempt_online_search = False
                 rag_answer = None # Initialize rag_answer outside the if/else

                 if current_doc_count == 0:
                     # Repo is empty, online search is the primary/only option for non-command
                     if not serpapi_key_present:
                          response = "Online search is required as the repository is empty, but SerpAPI key is missing. Please add SERPAPI_KEY to your .env file."
                          processing_status.update(label="Online search skipped (SerpAPI missing).", state="error") # State changed to error
                     else:
                         attempt_online_search = True
                         processing_status.update(label=f"No documents loaded. Attempting online search for concerts by '{query_text}'...", state="running")
                 else:
                     # Repo has documents, try RAG first
                     processing_status.update(label=f"Documents loaded. Searching RAG for '{query_text}' using {st.session_state.llm_provider.upper()}...", state="running")
                     rag_answer = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)

                     lowercased_rag_answer = rag_answer.strip().lower()
                     rag_found_nothing = any(phrase in lowercased_rag_answer for phrase in not_found_phrases_rag) or \
                                         len(lowercased_rag_answer) < 30

                     if rag_found_nothing:
                         # RAG found nothing, *now* consider online search as a fallback
                         if not serpapi_key_present:
                             response = rag_answer + "\n\nNote: RAG found no info in documents, and online search was skipped because the SerpAPI key is missing."
                             processing_status.update(label="Online search fallback skipped (SerpAPI missing).", state="complete") # State changed to complete
                         else:
                             attempt_online_search = True
                             processing_status.update(label=f"RAG found no specific info in documents. Attempting online search for '{query_text}' as a potential artist name...", state="running")
                     else:
                         # RAG found something, use the RAG answer
                         processing_status.update(label=f"RAG found relevant information.", state="complete")
                         response = rag_answer # response is already set here, no need to update it below


                 # Perform online search ONLY if the flag is set
                 if attempt_online_search:
                     response = perform_online_concert_search(query_text, llm_client, st.session_state.llm_provider)
                     # Check the response from online search in case it failed internally or found nothing
                     if response.startswith("Error:"):
                          processing_status.update(label="Online search failed.", state="error") # State changed to error
                     elif "No concert information found for " in response:
                          processing_status.update(label="Online search found nothing.", state="complete") # State changed to complete
                     else:
                          processing_status.update(label="Online search completed.", state="complete") # State changed to complete


        except Exception as e:
            logging.error(f"An unexpected error occurred during request processing: {e}", exc_info=True)
            response = f"An unexpected error occurred: {e}. Please check the application logs."
            # Ensure the status is updated here on general exception
            processing_status.update(label="An unexpected error occurred.", state="error")
            st.session_state.messages.append(("error", response)) # Add error specifically to history


    # Append bot response to history and redraw display
    st.session_state.messages.append(("bot", response))
    display_messages()

    # Note: The problematic line `st.session_state.user_input = ""` is correctly removed.
    # The input box value will persist until the next interaction or page refresh.


# --- Footer or additional info ---
st.sidebar.markdown("---")
st.sidebar.subheader("Repository Status")
if repository:
    st.sidebar.info(f"Documents Stored: {repository.get_total_documents()}")
else:
    st.sidebar.warning("Repository failed to load.")