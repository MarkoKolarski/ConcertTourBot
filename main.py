import sys
import logging
from dotenv import load_dotenv
from config import GEMINI_API_KEY
from repository_utils import get_repository
from document_processor import is_concert_domain, summarize_document
from qa_handler import answer_question
from llm_integrator import get_llm_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

def select_llm_provider() -> str:
    """
    Prompts the user to select an LLM provider and validates the choice.
    Google Gemini is recommended for the most accurate and reliable results.
    """

    GEMINI_OPTION = '1'
    HUGGINGFACE_OPTION = '2'
    EXIT_OPTION = '0'

    def print_provider_menu():
        print("Please choose which LLM provider you would like to use:\n")
        print("  1: Google Gemini (API) (Recommended)")
        print("     - Cloud-based model with high reliability and accuracy")
        print("     - Requires a (free) GOOGLE_API_KEY\n")
        print("  2: Hugging Face (Local)")
        print("     - Runs locally, lightweight and fast")
        print("     - May be less reliable and prone to hallucinations\n")
        print("  0: Exit the program\n")

    def handle_exit():
        print("\nExiting program. Goodbye!")
        sys.exit(0)

    def handle_gemini_selection():
        if not GEMINI_API_KEY:
            print("\nERROR: Google Gemini selected, but GOOGLE_API_KEY is not set.")
            print("Please set the API key in your .env file before using this option.")
            print("Alternatively, select the local Hugging Face model or exit the program.\n")
            return None
        print("\nYou have selected: Google Gemini")
        return 'gemini'

    def handle_huggingface_selection():
        print("\nYou have selected: Hugging Face (Local Model)")
        return 'huggingface'

    while True:
        print_provider_menu()
        choice = input("Enter your choice (0, 1, or 2): ").strip()

        if choice == EXIT_OPTION:
            handle_exit()

        elif choice == GEMINI_OPTION:
            provider = handle_gemini_selection()
            if provider:
                return provider

        elif choice == HUGGINGFACE_OPTION:
            return handle_huggingface_selection()

        else:
            print("\nInvalid input. Please enter 0, 1, or 2.")


def print_help():
    print("\nAvailable Commands:")
    print("  ADD: <document text>   - Add a new document about a concert tour.")
    print("  QUERY: <your question> - Ask a question about the concert tours.")
    print("  COUNT                  - Show the number of documents stored.")
    print("  PROVIDER               - Show the currently active LLM provider.")
    print("  CHANGE_PROVIDER        - Change the LLM provider.")
    print("  HELP                   - Show this help message.")
    print("  EXIT                   - Exit the service.")


def initialize_system():
    """Initialize LLM client and repository for the application."""
    logging.info("------------------------------------")
    logging.info(" Concert Tour Information Service ")
    logging.info("------------------------------------")

    # --- Interactive LLM Provider Selection ---
    chosen_provider_name = select_llm_provider()
    logging.info(f"User selected LLM provider: {chosen_provider_name.upper()}")

    try:
        # --- Initialize LLM Client ---
        print(f"\nInitializing LLM provider: {chosen_provider_name.upper()}...")
        llm_client, active_provider = get_llm_client(chosen_provider_name)

        if llm_client is None:
            print(f"\nFATAL ERROR: Failed to initialize LLM provider '{active_provider}'. Check logs and configuration.")
            logging.critical(f"Failed to get LLM client for provider: {active_provider}")
            sys.exit(1)

        logging.info(f"LLM Client for '{active_provider}' initialized.")

        # --- Initialize RAG Repository ---
        repository = get_repository()
        logging.info("RAG Repository initialized.")

        print(f"\nService Ready (Using {active_provider.upper()} LLM). Type 'HELP' for commands.")
        
        return llm_client, active_provider, repository
        
    except Exception as e:
        logging.critical(f"Fatal error during initialization: {e}", exc_info=True)
        print(f"\nFATAL ERROR: Could not initialize the service. Check logs. Error: {e}")
        sys.exit(1)


def handle_exit():
    """Handle EXIT command."""
    logging.info("Exit command received. Shutting down.")
    print("Exiting service. Goodbye!")
    return "EXIT"


def handle_help():
    """Display help information."""
    print_help()


def handle_show_provider(active_provider):
    """Show current LLM provider."""
    print(f"Currently active LLM Provider: {active_provider}")


def handle_change_provider(active_provider):
    """Change LLM provider."""
    logging.info("User requested to change LLM provider")
    print("Changing LLM provider...")

    new_provider_name = select_llm_provider()

    if new_provider_name == active_provider:
        print(f"Keeping the same provider: {active_provider}")
        return active_provider, None

    print(f"\nInitializing new LLM provider: {new_provider_name.upper()}...")
    new_llm_client, new_active_provider = get_llm_client(new_provider_name)

    if new_llm_client is None:
        logging.error(f"Failed to initialize new LLM provider: {new_provider_name}")
        print(f"Failed to change provider. Keeping current provider: {active_provider}")
        return active_provider, None

    logging.info(f"Successfully changed LLM provider to: {new_active_provider}")
    print(f"LLM provider successfully changed to: {new_active_provider.upper()}")
    return new_active_provider, new_llm_client


def handle_count_documents(repository):
    """Count documents in the repository."""
    count = repository.get_total_documents()
    print(f"Total documents currently stored: {count}")


def handle_add_document(document_text, llm_client, active_provider, repository):
    """Add document to the RAG repository."""
    if not document_text:
        logging.warning("ADD command received with empty document text.")
        print("Error: ADD command requires document text after 'ADD:'.")
        return

    logging.info("Processing ADD command...")

    if not is_concert_domain(document_text):
        logging.info("Document domain is not relevant.")
        print("\nSorry, I cannot ingest documents with other themes.")
        return

    print(f"Document seems relevant. Generating summary using {active_provider.upper()}...")
    summary = summarize_document(document_text, llm_client, active_provider)

    if summary.startswith("Error:"):
        print("\n--- Ingestion Failed ---")
        print(f"Could not process the document: {summary}")
        print("------------------------")
        return

    print("Summary generated. Ingesting into RAG system...")
    doc_id = repository.add_document_summary(summary)

    if doc_id == -1:
        logging.error("Failed to add generated summary to the RAG repository.")
        print("\n--- Ingestion Failed ---")
        print("Error: Generated summary but failed to save it.")
        print("------------------------")
        return

    logging.info(f"Document summary added to RAG with ID: {doc_id}")
    print("\n--- Ingestion Confirmation ---")
    print("Document successfully added.")
    print(f"LLM ({active_provider.upper()}) Summary:")
    print(f"'{summary}'")
    print("-----------------------------")


def handle_query(query_text, llm_client, active_provider, repository):
    """Process query against the RAG repository."""
    if not query_text:
        logging.warning("QUERY command received with empty query text.")
        print("Error: QUERY command requires a question after 'QUERY:'.")
        return

    logging.info(f"Processing QUERY command: '{query_text}'")
    print(f"Searching and generating answer using {active_provider.upper()}...")

    answer = answer_question(query_text, repository, llm_client, active_provider)

    print("\n--- Answer ---")
    print(answer)
    print("--------------")


def handle_unknown_command():
    """Handle unknown commands."""
    logging.warning("Unknown command received.")
    print("Error: Unknown command. Type 'HELP' for options.")


def process_command(command, user_input, llm_client, active_provider, repository):
    """Process user command and dispatch to appropriate handler."""
    # Extract base command and parameters
    command_parts = user_input.strip().split(':', 1)
    base_command = command_parts[0].upper()
    params = command_parts[1].strip() if len(command_parts) > 1 else ""
    
    # Commands without parameters
    command_handlers = {
        "HELP": lambda: handle_help(),
        "EXIT": lambda: handle_exit(),
        "PROVIDER": lambda: handle_show_provider(active_provider),
        "COUNT": lambda: handle_count_documents(repository),
        "CHANGE_PROVIDER": lambda: handle_change_provider(active_provider),
    }
    
    # Commands requiring parameters
    parameterized_handlers = {
        "ADD": lambda p: handle_add_document(p, llm_client, active_provider, repository),
        "QUERY": lambda p: handle_query(p, llm_client, active_provider, repository),
    }
    
    if base_command in command_handlers:
        result = command_handlers[base_command]()
        # Special handling for commands that return values
        if base_command == "EXIT":
            return result
        elif base_command == "CHANGE_PROVIDER" and isinstance(result, tuple):
            new_provider, new_client = result
            if new_client:
                return new_provider, new_client
    # Handle parameterized commands
    elif base_command in parameterized_handlers:
        if not params:
            print(f"Error: {base_command} command requires additional parameters.")
            return None
        parameterized_handlers[base_command](params)
    else:
        handle_unknown_command()
    
    return None


def main_loop():
    """Main application loop for the Concert Tour Information Service."""
    llm_client, active_provider, repository = initialize_system()
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()

            if not user_input:
                continue
                
            result = process_command(user_input, user_input, llm_client, active_provider, repository)
            
            if result == "EXIT":
                break
            elif isinstance(result, tuple):
                # Provider change
                active_provider, llm_client = result

        except EOFError:
            logging.info("EOF detected. Shutting down.")
            print("\nExiting service. Goodbye!")
            break

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt detected. Shutting down.")
            print("\nExiting service. Goodbye!")
            break

        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
            print(f"\nAn unexpected error occurred: {e}. Please check logs.")

if __name__ == "__main__":
    main_loop()