import sys
import logging
import traceback
from typing import Tuple, Any
from config import (
    HF_SUMMARIZATION_MODEL, HF_QA_MODEL, HF_MAX_INPUT_LENGTH,
    GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_SAFETY_SETTINGS, GEMINI_GENERATION_CONFIG,
    MAX_LLM_INPUT_CHARS
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _initialize_huggingface():
    """Loads Hugging Face pipelines. Returns pipeline dictionary."""
    try:
        from transformers import pipeline, AutoTokenizer
        logging.info(f"Initializing Hugging Face summarization pipeline with model: {HF_SUMMARIZATION_MODEL}")
        summarizer_tokenizer = AutoTokenizer.from_pretrained(HF_SUMMARIZATION_MODEL)
        summarizer = pipeline("summarization", model=HF_SUMMARIZATION_MODEL, tokenizer=summarizer_tokenizer)

        logging.info(f"Initializing Hugging Face text-generation pipeline for QA with model: {HF_QA_MODEL}")
        qa_tokenizer = AutoTokenizer.from_pretrained(HF_QA_MODEL)
        task = "text2text-generation" if "t5" in HF_QA_MODEL else "text-generation"
        qa_generator = pipeline(task, model=HF_QA_MODEL, tokenizer=qa_tokenizer)

        hf_pipelines = { # Create local dict to return
            "summarizer": summarizer,
            "qa_generator": qa_generator,
            "summarizer_tokenizer": summarizer_tokenizer,
            "qa_tokenizer": qa_tokenizer
            }
        logging.info("Hugging Face pipelines initialized successfully.")
        return hf_pipelines
    except ImportError:
        logging.error("Failed to import 'transformers'. Please install it: pip install transformers torch (or tensorflow)")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error initializing Hugging Face pipelines: {e}")
        sys.exit(1)

def _initialize_gemini():
    """Configures and returns the Google Gemini client."""

    if not GEMINI_API_KEY:
        logging.error("Gemini API Key is missing.")
        return None

    try:
        import google.generativeai as genai
        logging.info(f"Configuring Google Gemini with model: {GEMINI_MODEL_NAME}")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            safety_settings=GEMINI_SAFETY_SETTINGS,
            generation_config=genai.types.GenerationConfig(**GEMINI_GENERATION_CONFIG)
            )
        logging.info("Google Gemini configured successfully.")
        return gemini_model
    except ImportError:
        logging.error("Failed to import 'google.generativeai'. Please install it: pip install google-generativeai")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error configuring Google Gemini: {e}")
        return None

def get_llm_client(provider: str) -> Tuple[Any, str]:
    """
    Initializes and returns the LLM client based on the chosen provider.

    Args:
        provider: The chosen provider ('huggingface' or 'gemini').

    Returns:
        A tuple containing (client_object, provider_name).
        The client_object is a dictionary for 'huggingface' or a GenerativeModel for 'gemini'.
        Returns (None, provider) if initialization fails.
    """
    provider = provider.lower()
    if provider == 'huggingface':
        client = _initialize_huggingface()
        return client, provider
    elif provider == 'gemini':
        client = _initialize_gemini()
        return client, provider
    else:
        logging.error(f"Invalid LLM provider requested: {provider}")
        return None, provider

def _truncate_text(text: str, max_length: int, provider: str, tokenizer=None) -> str:
    """Truncates text based on provider specifics or char count."""
    if provider == 'huggingface' and tokenizer:
        # Use tokenizer for more accurate length check
        hf_max_len = max_length if max_length > 0 else HF_MAX_INPUT_LENGTH
        tokens = tokenizer.encode(text, truncation=False)
        if len(tokens) > hf_max_len:
            truncated_tokens = tokens[:hf_max_len]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            logging.warning(f"Input text truncated to {hf_max_len} tokens for Hugging Face model.")
            return truncated_text
        return text
    elif provider == 'gemini':
        # Gemini uses character count for truncation
         if len(text) > MAX_LLM_INPUT_CHARS:
             logging.warning(f"Input text truncated to {MAX_LLM_INPUT_CHARS} characters for Gemini.")
             return text[:MAX_LLM_INPUT_CHARS]
         return text
    else:
        if len(text) > MAX_LLM_INPUT_CHARS:
             logging.warning(f"Input text truncated to {MAX_LLM_INPUT_CHARS} characters (fallback).")
             return text[:MAX_LLM_INPUT_CHARS]
        return text

def generate_summary(text: str, client: Any, provider: str) -> str:
    """Generates a summary using the specified LLM client and provider."""
    if not client:
        return f"Error: LLM client for provider '{provider}' is not available."

    if provider == 'huggingface':
        try:
            pipeline = client['summarizer']
            tokenizer = client['summarizer_tokenizer']
            truncated_text = _truncate_text(text, HF_MAX_INPUT_LENGTH, provider, tokenizer)
            summary_result = pipeline(truncated_text, max_length=150, min_length=30, do_sample=False)
            return summary_result[0]['summary_text'].strip()
        except Exception as e:
            logging.error(f"Hugging Face summarization failed: {e}")
            return "Error: Could not generate summary using Hugging Face."

    elif provider == 'gemini':
        try:
            model = client
            truncated_text = _truncate_text(text, 0, provider)
            prompt = f"Summarize the following document about a concert tour:\n\n{truncated_text}\n\nSummary:"
            response = model.generate_content(prompt)
            # Robust checking for blocked content or empty response
            if not response.parts:
                 block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                 logging.warning(f"Gemini summarization blocked or empty: {block_reason}")
                 return f"Error: Summarization blocked by safety filters or empty response ({block_reason})."
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini summarization failed: {e}")
            return "Error: Could not generate summary using Gemini."
    else:
        return f"Error: Invalid LLM provider '{provider}' specified."


def generate_qa_answer(query: str, context: str, client: Any, provider: str) -> str:
    """Generates an answer to a query based on context using the specified LLM."""
    if not client:
        return f"Error: LLM client for provider '{provider}' is not available."
    
    prompt = f"""Context: {context}

Question: {query}

Answer:"""
    
    if provider == 'huggingface':
        try:
            pipeline = client['qa_generator']
            tokenizer = client['qa_tokenizer']
            
            # Check model type
            is_encoder_decoder = getattr(pipeline.model, "is_encoder_decoder", False)
            model_name = HF_QA_MODEL.lower()
            
            if is_encoder_decoder or "t5" in model_name or "bart" in model_name:
                input_text = f"question: {query} context: {context}"
                truncated_input = _truncate_text(input_text, HF_MAX_INPUT_LENGTH, provider, tokenizer)
                
                results = pipeline(
                    truncated_input, 
                    max_length=200, 
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
                answer = results[0]['generated_text']
            else:
                truncated_prompt = _truncate_text(prompt, HF_MAX_INPUT_LENGTH, provider, tokenizer)
                
                # Generate with clear stopping criteria
                results = pipeline(
                    truncated_prompt,
                    max_new_tokens=200,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=3
                )
                
                # Extract only the answer part
                full_text = results[0]['generated_text']
                answer_start = full_text.find("Answer:") + len("Answer:")
                
                if answer_start >= len("Answer:"):
                    answer = full_text[answer_start:].strip()
                else:
                    # If "Answer:" not found, take everything after the original prompt
                    answer = full_text[len(truncated_prompt):].strip()
            
            if not answer or len(answer) < 5:
                # Try a fallback approach - generate from scratch with a simpler prompt
                fallback_prompt = f"Based on this information: {context}\n\nAnswer this question: {query}"
                truncated_fallback = _truncate_text(fallback_prompt, HF_MAX_INPUT_LENGTH, provider, tokenizer)
                
                if is_encoder_decoder:
                    fallback_results = pipeline(
                        truncated_fallback,
                        max_length=200,
                        num_return_sequences=1
                    )
                    answer = fallback_results[0]['generated_text']
                else:
                    fallback_results = pipeline(
                        truncated_fallback,
                        max_new_tokens=200,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    answer = fallback_results[0]['generated_text'][len(truncated_fallback):].strip()
            
            if not answer or len(answer.strip()) < 5:
                return "Based on the available information, I couldn't generate a specific answer to your query. Please try asking in a different way."
                
            return answer.strip()
            
        except Exception as e:
            logging.error(f"Hugging Face QA generation failed: {e}")
            traceback_info = traceback.format_exc()
            logging.error(f"Traceback: {traceback_info}")
            return f"Error: Could not generate answer using Hugging Face."
    
    elif provider == 'gemini':
        try:
            model = client
            truncated_prompt = _truncate_text(prompt, 0, provider)
            response = model.generate_content(truncated_prompt)
            
            if not response.parts:
                block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
                logging.warning(f"Gemini QA blocked or empty: {block_reason}")
                return f"Error: Answer generation blocked by safety filters or empty response ({block_reason})."
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini QA generation failed: {e}")
            return "Error: Could not generate answer using Gemini."
    else:
        return f"Error: Invalid LLM provider '{provider}' specified."