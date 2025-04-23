import streamlit as st
import logging
from dotenv import load_dotenv
from typing import Any
import requests
import os
from dotenv import load_dotenv

from config import (
    GEMINI_API_KEY, GEMINI_API_KEY_ENV_VAR
)

from repository_utils import get_repository, ConcertRAGRepository
from document_processor import is_concert_domain, summarize_document
from qa_handler import answer_question
from llm_integrator import get_llm_client
from llm_integrator import generate_qa_answer

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

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
    
    # Get API key from environment variables
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return "SerpAPI key not found in environment variables. Please add SERPAPI_KEY to your .env file."
    
    logging.info(f"Performing online search for concerts by: {artist_name}")
    search_query = f"{artist_name} upcoming concerts tour dates 2025"
    
    try:
        # SerpAPI search
        url = "https://serpapi.com/search"
        params = {
            "q": search_query,
            "api_key": serpapi_key,
            "engine": "google"
        }
        
        logging.info(f"Sending request to SerpAPI for: {search_query}")
        response = requests.get(url, params=params)
        search_data = response.json()
        
        if "error" in search_data:
            logging.error(f"SerpAPI error: {search_data['error']}")
            return f"Error performing search: {search_data['error']}"
        
        # Extract relevant information from the search results
        search_results = []
        
        # Check for organic results
        if "organic_results" in search_data:
            for result in search_data["organic_results"][:5]:  # Limit to first 5 results
                if "snippet" in result:
                    search_results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    })
        
        # Check for answer box if available
        if "answer_box" in search_data and "snippet" in search_data["answer_box"]:
            search_results.insert(0, {
                "title": search_data["answer_box"].get("title", "Featured Snippet"),
                "snippet": search_data["answer_box"].get("snippet", ""),
                "link": search_data["answer_box"].get("link", "")
            })
        
        # Check for knowledge graph if available
        if "knowledge_graph" in search_data and "description" in search_data["knowledge_graph"]:
            search_results.insert(0, {
                "title": search_data["knowledge_graph"].get("title", artist_name),
                "snippet": search_data["knowledge_graph"].get("description", ""),
                "link": ""
            })
        
        if not search_results:
            logging.warning(f"No search results found for {artist_name} concerts")
            return f"No concert information found for {artist_name}. They may not have announced any upcoming concerts yet."
        
        context = f"Search results for '{artist_name} upcoming concerts':\n\n"
        for i, result in enumerate(search_results):
            context += f"Result {i+1}:\n"
            context += f"Title: {result['title']}\n"
            context += f"Snippet: {result['snippet']}\n"
            context += f"Link: {result['link']}\n\n"
        
        prompt = f"""
        Based on the following search results about {artist_name}'s upcoming concerts, 
        please provide a concise summary of:
        
        1. Any upcoming tour dates and venues mentioned
        2. Tour name (if any)
        3. Special guests or opening acts (if any)
        4. Ticket information (if available)
        
        If the search results don't mention specific concerts or tour dates, 
        please state that no specific concert information was found based on the search results.
        
        Be factual and only include information that's present in the search results.
        """
        
        # Generate answer using LLM
        logging.info(f"Sending search results to {provider_name} for synthesis")
        concert_info = generate_qa_answer(prompt, context, llm_client, provider_name)
        
        if concert_info.startswith("Error:"):
            logging.error(f"LLM error: {concert_info}")
            return f"Found information online, but couldn't synthesize a clear answer."
        
        return f"Information about {artist_name}'s upcoming concerts:\n\n{concert_info.strip()}"
        
    except Exception as e:
        logging.error(f"Error in concert search: {str(e)}", exc_info=True)
        return f"An error occurred while searching for concerts for {artist_name}: {str(e)}"


# --- Streamlit App Initialization ---

st.set_page_config(page_title="Concert Tour Info Bot", layout="wide")

st.title("Concert Tour Information Bot")
st.markdown("""
Welcome to the Concert Tour Info Bot!
You can **ADD** documents about concert tours or **QUERY** existing information.
If no documents are added to the repository, you can enter an **artist or band name** to search for their upcoming concerts online.
If documents are present, the bot will first try to answer from them (RAG). If no relevant information is found in your documents, it will attempt an online search for concerts by the artist/band name you provided.
""")

# --- LLM Provider Selection (using session state) ---
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
    index=list(provider_options.values()).index(st.session_state.llm_provider),
    key="provider_radio"
)
# Update session state based on user selection only if it changed
if st.session_state.llm_provider != provider_options[selected_provider_label]:
     st.session_state.llm_provider = provider_options[selected_provider_label]
     # Streamlit handles re-running and re-caching when this session state changes

# Display Gemini API key status
if st.session_state.llm_provider == 'gemini':
    if GEMINI_API_KEY:
        st.sidebar.success("Gemini API Key Loaded")
    else:
         st.sidebar.error(f"Gemini API Key ({GEMINI_API_KEY_ENV_VAR}) not found in .env")
         st.warning("Gemini API key not set. Please set it in your .env file to use the Gemini provider.")


# --- Cached Resources Initialization ---
# These functions will run only once per session or when dependencies change

@st.cache_resource(hash_funcs={str: id}) # Hash string argument by value
def get_llm_client_cached(provider_name: str) -> Any:
    """Caches the initialization of the LLM client."""
    st.info(f"Initializing {provider_name.upper()} LLM client...")
    try:
        # Use the get_llm_client function from your module
        client, actual_provider_name = get_llm_client(provider_name)
        if client is None:
             st.error(f"Failed to initialize LLM client for {provider_name}. Check logs and configuration.")
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

# Check if initialization failed - stop app if core components are missing
if llm_client is None or repository is None:
     st.error("Core components failed to initialize. Please check configuration and logs.")
     st.stop()


# --- Session State for Chat History ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
st.subheader("Interaction History")
# Using a markdown block to display history is simpler than st.empty() for appending
chat_history_placeholder = st.empty()

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

    # Update the placeholder content
    chat_history_placeholder.markdown(formatted_history)

# Initial display of messages
display_messages()


# --- User Input ---
# Using a key allows Streamlit to manage the input value in session_state automatically
# We removed the direct assignment to st.session_state.user_input = ""
user_input = st.text_input("Enter your request (e.g., 'ADD: <text>', 'QUERY: <question>', or Artist/Band Name):", key="user_input_widget")
process_button = st.button("Process Request")


# --- Process User Input ---
if process_button and user_input:
    # Append user message immediately to show it's received
    st.session_state.messages.append(("user", user_input))
    display_messages()

    response = ""
    user_input_upper = user_input.upper()
    query_text = user_input.strip() # Get the stripped input early

    with st.spinner("Processing..."):
        try:
            # 1. Handle Explicit Commands
            if user_input_upper.startswith("ADD:"):
                document_text = user_input[4:].strip()
                if not document_text:
                    response = "Error: ADD command requires document text after 'ADD:'."
                elif is_concert_domain(document_text):
                    st.session_state.messages.append(("info", "Document seems relevant. Generating summary..."))
                    display_messages()
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
                 query_text_from_command = user_input[6:].strip()
                 if not query_text_from_command:
                     response = "Error: QUERY command requires a question after 'QUERY:'."
                 else:
                    st.session_state.messages.append(("info", f"Searching RAG for '{query_text_from_command}' using {st.session_state.llm_provider.upper()} (Explicit QUERY)..."))
                    display_messages()
                    response = answer_question(query_text_from_command, repository, llm_client, st.session_state.llm_provider)

            elif user_input_upper == "COUNT":
                 count = repository.get_total_documents()
                 response = f"Total documents currently stored: {count}"

            elif user_input_upper == "HELP":
                 response = """
Available Commands:
ADD: <document text>   - Add a new document about a concert tour.
QUERY: <your question> - Ask a question about the concert tours (RAG).
COUNT                  - Show the number of documents stored (RAG).
<Artist or Band Name>  - If no documents are loaded, or RAG finds no info in documents, attempt online search.
"""
            # 2. Handle Non-Command Input (Potential RAG Query with Online Search Fallback)
            else:
                 current_doc_count = repository.get_total_documents()

                 # Define phrases that indicate RAG found no info (case-insensitive)
                 # These should match the messages potentially returned by qa_handler.answer_question
                 not_found_phrases_rag = [
                     "couldn't find specific information related to your query in the ingested documents",
                     "i couldn't find specific information", # From a previous iteration
                     "based on the available information, i couldn't generate", # From a previous iteration
                     "no information about", # From a previous iteration
                     "no relevant summaries found", # From a previous iteration
                 ]

                 # If repository is empty, skip RAG and go directly to online search
                 if current_doc_count == 0:
                     st.session_state.messages.append(("info", f"No documents loaded. Attempting online search for concerts by '{query_text}'..."))
                     display_messages()
                     response = perform_online_concert_search(query_text, llm_client, st.session_state.llm_provider)

                 else:
                     # Repository has documents, try RAG first
                     st.session_state.messages.append(("info", f"Documents loaded. Searching RAG for '{query_text}' using {st.session_state.llm_provider.upper()}..."))
                     display_messages()
                     rag_answer = answer_question(query_text, repository, llm_client, st.session_state.llm_provider)

                     # Check if the RAG answer strongly suggests no relevant info was found in the documents
                     lowercased_rag_answer = rag_answer.strip().lower()
                     rag_found_nothing = any(phrase in lowercased_rag_answer for phrase in not_found_phrases_rag) or \
                                         len(lowercased_rag_answer) < 30 # Also consider very short answers as potentially "not found"

                     if rag_found_nothing:
                         # RAG found nothing, *now* try online search as a fallback
                         st.session_state.messages.append(("info", f"RAG found no specific info in documents. Attempting online search for '{query_text}' as a potential artist name..."))
                         display_messages()
                         response = perform_online_concert_search(query_text, llm_client, st.session_state.llm_provider)
                     else:
                         # RAG found something, use the RAG answer
                         st.session_state.messages.append(("info", f"RAG found relevant information."))
                         display_messages()
                         response = rag_answer


        except Exception as e:
            logging.error(f"An error occurred during request processing: {e}", exc_info=True)
            response = f"An unexpected error occurred: {e}. Please check the application logs."
            st.session_state.messages.append(("error", response)) # Add error specifically

    # Append bot response to history and update display
    st.session_state.messages.append(("bot", response))
    display_messages()

    # Note: The problematic line `st.session_state.user_input = ""` is correctly removed.
    # The input box value will persist until the next interaction or page refresh.


# Optional: Display repository status in sidebar
st.sidebar.markdown("---")
if repository:
    st.sidebar.info(f"Repository Status:\nDocuments: {repository.get_total_documents()}")
else:
    st.sidebar.warning("Repository failed to load.")