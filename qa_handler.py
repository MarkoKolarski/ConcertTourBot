import logging
from typing import Any # Added for type hinting
from repository_utils import ConcertRAGRepository
from config import QA_TOP_K
from llm_integrator import generate_qa_answer # Import remains the same

# --- Updated function signature ---
def answer_question(
    query: str,
    repository: ConcertRAGRepository,
    llm_client: Any,
    provider_name: str
) -> str:
    """
    Answers a user's question based on information retrieved from the repository,
    using the specified LLM client to synthesize the final answer.

    Args:
        query: The user's question.
        repository: The initialized ConcertRAGRepository instance.
        llm_client: The initialized LLM client object (HF dict or Gemini model).
        provider_name: The name of the active provider ('huggingface' or 'gemini').


    Returns:
        A string containing the answer, grounded in the retrieved documents,
        or an error/default message if generation fails or no info is found.
    """
    logging.info(f"Answering query using LLM provider {provider_name}: '{query}'")
    relevant_summaries = repository.search_relevant_summaries(query, k=QA_TOP_K)

    if not relevant_summaries:
        logging.warning("No relevant summaries found for the query.")
        return "I couldn't find specific information related to your query in the ingested documents."

    context = "\n\n---\n\n".join(relevant_summaries)
    logging.info(f"Context for LLM QA ({provider_name}):\n{context[:500]}...")

    # Call the LLM integration function, passing the client and provider
    answer = generate_qa_answer(query, context, llm_client, provider_name)

    if answer.startswith("Error:"):
        logging.error(f"QA generation failed ({provider_name}): {answer}")
    else:
        logging.info(f"Answer generated successfully ({provider_name}).")

    return answer