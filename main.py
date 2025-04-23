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
    """Prompts the user to select an LLM provider and validates the choice."""
    while True:
        print("\nPlease select the LLM provider:")
        print("  1: Hugging Face (Local Models - Requires 'transformers' and 'torch'/'tensorflow')")
        print("  2: Google Gemini (Cloud API - Requires GOOGLE_API_KEY in .env)")
        print("  0: Exit the program")
        choice = input("Enter choice (0, 1, or 2): ").strip()
        
        if choice == '0':
            print("Exiting program...")
            sys.exit(0)
        elif choice == '1':
            print("Selected: Hugging Face")
            return 'huggingface'
        elif choice == '2':
            if not GEMINI_API_KEY:
                print("\nERROR: Google Gemini selected, but GOOGLE_API_KEY is not set in your .env file.")
                print("Please set the API key, choose Hugging Face, or exit.")
            else:
                print("Selected: Google Gemini")
                return 'gemini'
        else:
            print("Invalid choice. Please enter 0, 1, or 2.")


def print_help():
    print("\nAvailable Commands:")
    print("  ADD: <document text>   - Add a new document about a concert tour.")
    print("  QUERY: <your question> - Ask a question about the concert tours.")
    print("  COUNT                  - Show the number of documents stored.")
    print("  PROVIDER               - Show the currently active LLM provider.")
    print("  HELP                   - Show this help message.")
    print("  EXIT                   - Exit the service.")


def main_loop():
    logging.info("------------------------------------")
    logging.info(" Concert Tour Information Service ")
    logging.info("------------------------------------")

    # --- Interactive LLM Provider Selection ---
    chosen_provider_name = select_llm_provider()
    logging.info(f"User selected LLM provider: {chosen_provider_name.upper()}")

    try:
        # --- Initialize LLM Client (based on selection) ---
        print(f"\nInitializing LLM provider: {chosen_provider_name.upper()}...")
        llm_client, active_provider = get_llm_client(chosen_provider_name)

        if llm_client is None:
             print(f"\nFATAL ERROR: Failed to initialize LLM provider '{active_provider}'. Check logs and configuration.")
             logging.critical(f"Failed to get LLM client for provider: {active_provider}")
             sys.exit(1)

        logging.info(f"LLM Client for '{active_provider}' initialized.")
        print("LLM Client initialized.")


        # --- Initialize RAG Repository ---
        print("Initializing RAG Repository...")
        repository = get_repository()
        logging.info("RAG Repository initialized.")

        print(f"\nService Ready (Using {active_provider.upper()} LLM). Type 'HELP' for commands.")

    except Exception as e:
        logging.critical(f"Fatal error during initialization: {e}", exc_info=True)
        print(f"\nFATAL ERROR: Could not initialize the service. Check logs. Error: {e}")
        sys.exit(1)


    # --- Main Command Loop ---
    while True:
        try:
            user_input = input("\nEnter command: ").strip()

            if not user_input:
                continue

            if user_input.upper() == "EXIT":
                logging.info("Exit command received. Shutting down.")
                print("Exiting service. Goodbye!")
                break

            elif user_input.upper() == "HELP":
                print_help()
                continue

            elif user_input.upper() == "PROVIDER":
                print(f"Currently active LLM Provider: {active_provider}")
                continue

            elif user_input.upper() == "COUNT":
                count = repository.get_total_documents()
                print(f"Total documents currently stored: {count}")
                continue

            elif user_input.upper().startswith("ADD:"):
                document_text = user_input[4:].strip()
                if not document_text:
                    logging.warning("ADD command received with empty document text.")
                    print("Error: ADD command requires document text after 'ADD:'.")
                    continue

                logging.info("Processing ADD command...")
                if is_concert_domain(document_text):
                    logging.info("Document domain is relevant (Concert Tour).")
                    print(f"Document seems relevant. Generating summary using {active_provider.upper()}...")
                    summary = summarize_document(document_text, llm_client, active_provider)

                    if summary.startswith("Error:"):
                        print(f"\n--- Ingestion Failed ---")
                        print(f"Could not process the document: {summary}")
                        print("------------------------")
                    else:
                        print("Summary generated. Ingesting into RAG system...")
                        doc_id = repository.add_document_summary(summary)

                        if doc_id != -1:
                            logging.info(f"Document summary added to RAG with ID: {doc_id}")
                            print("\n--- Ingestion Confirmation ---")
                            print("Document successfully added.")
                            print(f"LLM ({active_provider.upper()}) Summary:")
                            print(f"'{summary}'")
                            print("-----------------------------")
                        else:
                            logging.error("Failed to add generated summary to the RAG repository.")
                            print("\n--- Ingestion Failed ---")
                            print("Error: Generated summary but failed to save it.")
                            print("------------------------")
                else:
                    logging.info("Document domain is not relevant.")
                    print("\nSorry, I cannot ingest documents with other themes.")

            elif user_input.upper().startswith("QUERY:"):
                query_text = user_input[6:].strip()
                if not query_text:
                    logging.warning("QUERY command received with empty query text.")
                    print("Error: QUERY command requires a question after 'QUERY:'.")
                    continue

                logging.info(f"Processing QUERY command: '{query_text}'")
                print(f"Searching and generating answer using {active_provider.upper()}...")
                answer = answer_question(query_text, repository, llm_client, active_provider)

                print("\n--- Answer ---")
                print(answer)
                print("--------------")

            else:
                logging.warning(f"Unknown command received: {user_input}")
                print("Error: Unknown command. Type 'HELP' for options.")

        # --- Exception Handling for the Loop ---
        except EOFError:
             logging.info("EOF detected. Shutting down.")
             print("\nExiting service. Goodbye!")
             break
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt detected. Shutting down.")
            print("\nExiting service. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print(f"\nAn unexpected error occurred: {e}. Please check logs.")


if __name__ == "__main__":
    main_loop()