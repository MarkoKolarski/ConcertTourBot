import streamlit as st
import logging
from dotenv import load_dotenv
from typing import Any
import requests
import os
from repository_utils import get_repository, ConcertRAGRepository
from document_processor import is_concert_domain, summarize_document
from qa_handler import answer_question
from llm_integrator import get_llm_client, generate_qa_answer

from config import (
    GEMINI_API_KEY, GEMINI_API_KEY_ENV_VAR
)

DEFAULT_PROVIDER = 'gemini'
PROVIDER_OPTIONS = {
    'Google Gemini (API)': 'gemini',
    'Hugging Face (Local)': 'huggingface'
}
PAGE_TITLE = "Concert Bot 🎶"
SIDEBAR_TITLE = "Settings"
CHAT_CONTAINER_HEIGHT = 400
GEMINI_KEY_STATUS_LABEL = "Gemini API Key"
SERPAPI_KEY_STATUS_LABEL = "SerpAPI Key"

def _fetch_serpapi_results(query: str, api_key: str) -> dict:
    """Fetches search results from SerpAPI."""
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _extract_search_results(data: dict, artist_name: str) -> list:
    """Extracts relevant search results from SerpAPI response."""
    results = []

    # Answer Box
    answer_box = data.get("answer_box", {})
    if answer_box.get("snippet"):
        results.append({
            "title": answer_box.get("title", "Featured Snippet"),
            "snippet": answer_box.get("snippet", ""),
            "link": answer_box.get("link", "")
        })

    # Knowledge Graph
    knowledge_graph = data.get("knowledge_graph", {})
    if knowledge_graph.get("description"):
        results.append({
            "title": knowledge_graph.get("title", artist_name),
            "snippet": knowledge_graph.get("description", ""),
            "link": knowledge_graph.get("link", "")
        })

    # Organic Results
    for result in data.get("organic_results", [])[:5]:  # Limit to 5
        if "snippet" in result:
            results.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", "")
            })

    return results


def _build_llm_context(results: list, artist_name: str) -> str:
    """Formats results into a readable context string for LLM."""
    context = f"Search results for '{artist_name} upcoming concerts':\n\n"
    for i, result in enumerate(results):
        context += f"--- Result {i+1} ---\n"
        if result.get("title"):
            context += f"Title: {result['title']}\n"
        if result.get("snippet"):
            context += f"Snippet: {result['snippet']}\n"
        if result.get("link"):
            context += f"Link: {result['link']}\n"
        context += "\n"

    max_len = 3500
    if len(context) > max_len:
        logging.warning(f"Search context truncated to {max_len} characters.")
        context = context[:max_len] + "\n... (context truncated)"

    return context


def _validate_concert_response(response: str, artist_name: str) -> str:
    """Validates LLM response and returns a user-facing message."""
    if response.startswith("Error:"):
        logging.error(f"LLM synthesis error: {response}")
        return f"Found information online, but couldn't synthesize a clear answer: {response}"

    cleaned = response.strip().lower()

    # Check for vague/generic answers
    vague_phrases = [
        "could not find",
        "based on the available information",
        "i couldn't generate a specific answer",
        "no specific dates",
        "no concert information was found"
    ]

    if not cleaned or len(cleaned) < 30 or any(phrase in cleaned for phrase in vague_phrases):
        return f"Found online information for {artist_name}, but the bot could not synthesize specific upcoming concert dates from the search results."

    return f"Information about {artist_name}'s upcoming concerts:\n\n{response.strip()}"


def perform_online_concert_search(artist_name: str, llm_client: Any, provider_name: str) -> str:
    """
    Performs an online search for upcoming concerts for a given artist
    and uses the LLM to synthesize an answer from the search results via SerpAPI.
    """
    if not artist_name:
        return "Please provide a musician or band name to search for concerts."

    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        logging.error("perform_online_concert_search called without SerpAPI key present.")
        return "Error: SerpAPI key is missing, cannot perform online search."

    logging.info(f"Performing online search for concerts by: {artist_name}")
    search_query = f"{artist_name} upcoming concerts tour dates 2025"

    try:
        search_data = _fetch_serpapi_results(search_query, serpapi_key)
        if "error" in search_data:
            return f"Error performing search: {search_data['error']}"

        search_results = _extract_search_results(search_data, artist_name)

        if not search_results:
            logging.warning(f"No search results found for {artist_name} concerts")
            return f"No concert information found for {artist_name}. They may not have announced any upcoming concerts yet or the search did not return relevant results."

        context = _build_llm_context(search_results, artist_name)
        llm_query = f"Summarize upcoming concert details for {artist_name} based *only* on the provided search results."

        logging.info(f"Sending search results to {provider_name} for synthesis")
        concert_info = generate_qa_answer(llm_query, context, llm_client, provider_name)

        return _validate_concert_response(concert_info, artist_name)

    except requests.exceptions.RequestException as req_err:
        logging.error(f"SerpAPI request error: {req_err}", exc_info=True)
        return f"Network or request error during search for {artist_name}: {str(req_err)}. Check your SerpAPI key and network connection."

    except Exception as e:
        logging.error(f"Unexpected error in concert search: {str(e)}", exc_info=True)
        return f"An unexpected error occurred while searching for concerts for {artist_name}: {str(e)}. Please check the logs for details."


def display_welcome_message():
    """Display welcome message and usage instructions."""
    with st.container(border=True):
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
        st.markdown("---")
        st.markdown("#### 🔑 API Key Requirements")
        st.markdown(f"""
        This application uses external services that require API keys for full functionality:
        - **`{GEMINI_API_KEY_ENV_VAR}`**: Required for the **Google Gemini** LLM provider. If this key is missing, you will only be able to use the Hugging Face (Local) provider.
        - **`SERPAPI_KEY`**: Required for the **Online Concert Search** functionality. If this key is missing, the bot cannot perform web searches for concert dates when needed.

        Please add these keys to your `.env` file located in the project's root directory.
        """)


def display_api_key_status():
    """Display the status of required API keys in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Status")

    if GEMINI_API_KEY:
        st.sidebar.success(f"✅ {GEMINI_KEY_STATUS_LABEL} Loaded")
    else:
        st.sidebar.warning(f"⚠️ {GEMINI_KEY_STATUS_LABEL} ({GEMINI_API_KEY_ENV_VAR}) not found")

    if os.getenv("SERPAPI_KEY"):
        st.sidebar.success("✅ SerpAPI Key Loaded (Online Search Enabled)")
    else:
        st.sidebar.warning("⚠️ SerpAPI Key not found")


@st.cache_resource(hash_funcs={str: id})
def get_llm_client_cached(provider_name: str) -> Any:
    """Initialize and cache the LLM client."""
    if provider_name == 'gemini' and not GEMINI_API_KEY:
        logging.error(f"Attempted to initialize Gemini LLM, but {GEMINI_API_KEY_ENV_VAR} is missing.")
        st.error(f"Cannot initialize Google Gemini LLM. {GEMINI_API_KEY_ENV_VAR} is not set.")
        return None

    st.info(f"Initializing {provider_name.upper()} LLM client...")
    try:
        client, actual_provider = get_llm_client(provider_name)
        if client is None:
            st.error(f"Failed to initialize LLM client for {provider_name}. Check logs.")
            return None

        st.success(f"{actual_provider.upper()} LLM client initialized.")
        return client
    except Exception as e:
        logging.error(f"Error initializing LLM client ({provider_name}): {e}", exc_info=True)
        st.error(f"Error initializing LLM client ({provider_name}): {e}")
        return None


@st.cache_resource
def get_repository_cached() -> ConcertRAGRepository:
    """Initialize and cache the RAG repository."""
    st.info("Initializing RAG Repository (FAISS Index and Summary Map)...")
    try:
        repo = get_repository()
        st.success("RAG Repository initialized.")
        return repo
    except Exception as e:
        logging.error(f"Error initializing RAG Repository: {e}", exc_info=True)
        st.error(f"Error initializing RAG Repository: {e}")
        return None


def initialize_llm_and_repo():
    """Initialize both LLM client and repository with visual feedback."""
    with st.status(f"Initializing {st.session_state.llm_provider.upper()} LLM...", expanded=True) as status_llm:
        llm_client = get_llm_client_cached(st.session_state.llm_provider)
        if not llm_client:
            status_llm.update(label=f"Failed to initialize {st.session_state.llm_provider.upper()} LLM.", state="error", expanded=True)
            st.stop()
        status_llm.update(label=f"{st.session_state.llm_provider.upper()} LLM initialized.", state="complete", expanded=False)

    with st.status("Initializing RAG Repository...", expanded=True) as status_repo:
        repository = get_repository_cached()
        if not repository:
            status_repo.update(label="Failed to initialize RAG Repository.", state="error", expanded=True)
            st.error("Fatal Error: Could not initialize the RAG Repository.")
            st.stop()
        doc_count = repository.get_total_documents()
        status_repo.update(label=f"RAG Repository initialized. Documents loaded: {doc_count}", state="complete", expanded=False)

    return llm_client, repository


def initialize_provider_selection():
    """Initialize and manage the LLM provider selection in the sidebar."""

    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = DEFAULT_PROVIDER

    # Determine the index of the current provider for the radio button
    try:
        default_index = list(PROVIDER_OPTIONS.values()).index(st.session_state.llm_provider)
    except ValueError:
        default_index = 0 

    # Render radio buttons
    selected_label = st.sidebar.radio(
        "Select LLM Provider:",
        options=list(PROVIDER_OPTIONS.keys()),
        index=default_index,
        key="provider_radio"
    )

    # Update session state if changed
    selected_value = PROVIDER_OPTIONS[selected_label]
    if st.session_state.llm_provider != selected_value:
        st.session_state.llm_provider = selected_value


def initialize_chat_session():
    """Initialize chat session state if not already present."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def display_messages(chat_container=None):
    """Display chat history in the provided container or directly in the app."""
    if chat_container:
        chat_container.empty()
        container = chat_container
    else:
        container = st

    with container:
        for msg_type, message in st.session_state.messages:
            if msg_type == "user":
                with st.chat_message("user"):
                    st.markdown(message)
            elif msg_type == "bot":
                with st.chat_message("assistant"):
                    st.markdown(message)
            elif msg_type == "info":
                st.info(f"ℹ️ {message}")
            elif msg_type == "error":
                st.error(f"❌ {message}")


def render_chat_input():
    """Render the user input area and process button."""
    input_container = st.container()
    with input_container:
        user_input = st.text_input(
            "Enter your request:",
            placeholder="e.g., 'ADD: document text', 'QUERY: question', or 'Billie Eilish'",
            key="user_input_widget",
            label_visibility="collapsed"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            process_button = st.button("Send Request", use_container_width=True)

    return user_input, process_button


# --- Command Handler Functions ---

def handle_add_command(user_input, llm_client, repository, processing_status):
    """Handle the ADD command to ingest documents."""
    document_text = user_input[4:].strip()
    
    if not document_text:
        processing_status.update(label="ADD command invalid.", state="error")
        return "Error: ADD command requires document text after 'ADD:'."
        
    if not is_concert_domain(document_text):
        processing_status.update(label="Document not relevant.", state="complete")
        return "Sorry, I cannot ingest documents with other themes."
    
    processing_status.update(label="Document seems relevant. Generating summary...", state="running")
    summary = summarize_document(document_text, llm_client, st.session_state.llm_provider)
    
    if summary.startswith("Error:"):
        processing_status.update(label="Summary generation failed.", state="error")
        return f"Could not process the document: {summary}"
    
    processing_status.update(label="Summary generated. Ingesting into RAG system...", state="running")
    doc_id = repository.add_document_summary(summary)
    
    if doc_id != -1:
        processing_status.update(label="Document successfully added.", state="complete")
        return f"Document successfully added (ID: {doc_id}).\n\nSummary:\n'{summary}'"
    else:
        processing_status.update(label="Document saving failed.", state="error")
        return f"Error: Generated summary but failed to save it.\n\nSummary:\n'{summary}'"


def handle_query_command(user_input, repository, llm_client, processing_status):
    """Handle the QUERY command to search RAG."""
    query_text = user_input[6:].strip()
    
    if not query_text:
        processing_status.update(label="QUERY command invalid.", state="error")
        return "Error: QUERY command requires a question after 'QUERY:'."
    
    processing_status.update(
        label=f"Searching RAG for '{query_text}' using {st.session_state.llm_provider.upper()}...", 
        state="running"
    )
    response = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)
    processing_status.update(label="RAG query processed.", state="complete")
    
    return response


def handle_count_command(repository, processing_status):
    """Handle the COUNT command to show document count."""
    count = repository.get_total_documents()
    processing_status.update(label="Document count retrieved.", state="complete")
    return f"Total documents currently stored: {count}"


def handle_help_command(processing_status):
    """Handle the HELP command to display available commands."""
    processing_status.update(label="Help information displayed.", state="complete")
    return """
Available Commands:
- **ADD:** `<document text>`: Add a new document about a concert tour.
- **QUERY:** `<your question>`: Ask a question based on added documents.
- **COUNT**: See how many documents are stored.
- Enter **Artist or Band Name**: Search online for concerts if RAG finds no info or repo is empty.
"""


def handle_empty_repository(query_text, serpapi_key_present, llm_client, processing_status):
    """Handle queries when repository is empty."""
    if not serpapi_key_present:
        processing_status.update(label="Online search skipped (SerpAPI missing).", state="error")
        return "Online search is required as the repository is empty, but SerpAPI key is missing. Please add SERPAPI_KEY to your .env file."
    
    processing_status.update(
        label=f"No documents loaded. Attempting online search for concerts by '{query_text}'...", 
        state="running"
    )
    response = perform_online_concert_search(query_text, llm_client, st.session_state.llm_provider)
    
    update_status_based_on_search_result(response, processing_status)
    return response


def handle_repository_search(query_text, repository, llm_client, not_found_phrases_rag, serpapi_key_present, processing_status):
    """Search repository and potentially fallback to online search."""
    processing_status.update(
        label=f"Documents loaded. Searching RAG for '{query_text}' using {st.session_state.llm_provider.upper()}...", 
        state="running"
    )
    rag_answer = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)
    
    if rag_found_nothing(rag_answer, not_found_phrases_rag):
        return handle_rag_no_results(rag_answer, query_text, serpapi_key_present, llm_client, processing_status)
    else:
        processing_status.update(label="RAG found relevant information.", state="complete")
        return rag_answer


def rag_found_nothing(rag_answer, not_found_phrases_rag):
    """Check if RAG found no relevant information."""
    lowercased_rag_answer = rag_answer.strip().lower()
    return any(phrase in lowercased_rag_answer for phrase in not_found_phrases_rag) or len(lowercased_rag_answer) < 30


def handle_rag_no_results(rag_answer, query_text, serpapi_key_present, llm_client, processing_status):
    """Handle case when RAG finds no results."""
    if not serpapi_key_present:
        processing_status.update(label="Online search fallback skipped (SerpAPI missing).", state="complete")
        return rag_answer + "\n\nNote: RAG found no info in documents, and online search was skipped because the SerpAPI key is missing."
    
    processing_status.update(
        label=f"RAG found no specific info in documents. Attempting online search for '{query_text}' as a potential artist name...", 
        state="running"
    )
    response = perform_online_concert_search(query_text, llm_client, st.session_state.llm_provider)
    
    update_status_based_on_search_result(response, processing_status)
    return response


def update_status_based_on_search_result(response, processing_status):
    """Update status based on the search result."""
    if response.startswith("Error:"):
        processing_status.update(label="Online search failed.", state="error")
    elif "No concert information found for " in response:
        processing_status.update(label="Online search found nothing.", state="complete")
    else:
        processing_status.update(label="Online search completed.", state="complete")


def process_user_request(user_input, user_input_upper, query_text, repository, llm_client, processing_status):
    """
    Processes user requests and generates appropriate responses
    using the match...case statement (Python 3.10+).
    """

    match user_input_upper:
        case cmd if cmd.startswith("ADD:"):
            return handle_add_command(user_input, llm_client, repository, processing_status)

        case cmd if cmd.startswith("QUERY:"):
            return handle_query_command(user_input, repository, llm_client, processing_status)

        case "COUNT":
            return handle_count_command(repository, processing_status)

        case "HELP":
            return handle_help_command(processing_status)

        case _:
            return handle_general_query(query_text, repository, llm_client, processing_status)


def handle_general_query(query_text, repository, llm_client, processing_status):
    """Handle non-command input as potential RAG query with online search fallback."""
    current_doc_count = repository.get_total_documents()
    serpapi_key_present = os.getenv("SERPAPI_KEY")

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

    # Check if repository is empty
    if current_doc_count == 0:
        return handle_empty_repository(query_text, serpapi_key_present, llm_client, processing_status)
    else:
        return handle_repository_search(
            query_text, 
            repository, 
            llm_client, 
            not_found_phrases_rag, 
            serpapi_key_present, 
            processing_status
        )


# --- MAIN APPLICATION CODE ---

# Setup logging and environment
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# Configure page
st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
# apply_custom_styles()

# Display page title and welcome message
st.title(PAGE_TITLE)
display_welcome_message()

# Configure sidebar and display API key status
st.sidebar.header(SIDEBAR_TITLE)
display_api_key_status()
initialize_provider_selection()

# Initialize LLM + Repository with feedback
llm_client, repository = initialize_llm_and_repo()

# Initialize session state for chat
initialize_chat_session()

# Display chat history in a scrollable container
st.subheader("Interaction History")
chat_container = st.container(height=CHAT_CONTAINER_HEIGHT)
display_messages(chat_container)

# Render user input area and return input/button state
user_input, process_button = render_chat_input()

# Process user input if button is clicked
if process_button and user_input:
    # Append and display user message
    st.session_state.messages.append(("user", user_input))
    display_messages(chat_container)

    response = ""
    query_text = user_input.strip()
    user_input_upper = user_input.upper()

    # Process the user input and generate response
    with st.status("Processing request...", expanded=True) as processing_status:
        try:
            response = process_user_request(
                user_input, 
                user_input_upper, 
                query_text, 
                repository, 
                llm_client, 
                processing_status
            )
        except Exception as e:
            logging.error(f"An unexpected error occurred during request processing: {e}", exc_info=True)
            response = f"An unexpected error occurred: {e}. Please check the application logs."
            processing_status.update(label="An unexpected error occurred.", state="error")
            st.session_state.messages.append(("error", response))

    # Append bot response to history and redraw display
    st.session_state.messages.append(("bot", response))
    display_messages(chat_container)

# Display repository status in sidebar footer
st.sidebar.markdown("---")
st.sidebar.subheader("Repository Status")
if repository:
    st.sidebar.info(f"Documents Stored: {repository.get_total_documents()}")
else:
    st.sidebar.warning("Repository failed to load.")