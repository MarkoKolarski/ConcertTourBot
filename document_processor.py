import logging
from typing import Any # Added for type hinting
from config import CONCERT_KEYWORDS
from llm_integrator import generate_summary # Import remains the same

def is_concert_domain(text: str) -> bool:
    """
    Checks if the document text likely belongs to the concert tour domain.
    (No changes needed here)
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CONCERT_KEYWORDS)

# --- Updated function signature ---
def summarize_document(text: str, llm_client: Any, provider_name: str) -> str:
    """
    Generates a concise summary of the document text using the specified LLM client.

    Args:
        text: The input document text.
        llm_client: The initialized LLM client object (HF dict or Gemini model).
        provider_name: The name of the active provider ('huggingface' or 'gemini').

    Returns:
        A summary string, or an error message if summarization fails.
    """
    logging.info(f"Attempting to generate summary using LLM provider: {provider_name}")
    if not text:
        logging.warning("Summarization attempt on empty text.")
        return "Error: Cannot summarize empty document."

    # Call the LLM integration function, passing the client and provider
    summary = generate_summary(text, llm_client, provider_name)

    if summary.startswith("Error:"):
        logging.error(f"Summarization failed ({provider_name}): {summary}")
    else:
        logging.info(f"Summary generated successfully ({provider_name}).")

    return summary