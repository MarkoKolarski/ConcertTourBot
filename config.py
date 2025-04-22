import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file into environment

# --- General Service Configuration ---
# Choose the LLM provider: 'huggingface' (default) or 'gemini'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", 'huggingface').lower()

# --- RAG System Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "concert_tour_index.faiss"
SUMMARY_MAPPING_PATH = "summary_mapping.json"
QA_TOP_K = 3

# --- Domain Checking Configuration ---
CONCERT_KEYWORDS = [
    "concert", "tour", "live music", "performance", "show", "gig",
    "venue", "arena", "stadium", "festival", "band", "artist",
    "musician", "singer", "schedule", "dates", "tickets", "logistics",
    "setlist", "headliner", "support act", "guest appearance"
]

# --- Hugging Face Configuration (Used if LLM_PROVIDER='huggingface') ---
# Recommended models:
# Summarization: 'facebook/bart-large-cnn', 'google/pegasus-xsum', 'sshleifer/distilbart-cnn-12-6' (smaller)
# QA/Generation: 'google-t5/t5-base', 'distilgpt2' (smaller, requires careful prompting)
# Note: Larger models require more RAM/VRAM and time.
HF_SUMMARIZATION_MODEL = os.getenv("HF_SUMMARIZATION_MODEL", 'sshleifer/distilbart-cnn-6-6')
HF_QA_MODEL = os.getenv("HF_QA_MODEL", 't5-small') # T5 is good for text-to-text tasks like QA


HF_MAX_INPUT_LENGTH = 1024 # Check model card for specific limits (e.g., BART: 1024, T5: 512 usually)

# --- Gemini Configuration (Used if LLM_PROVIDER='gemini') ---
GEMINI_API_KEY_ENV_VAR = "GOOGLE_API_KEY" # Name of the env variable holding the key
GEMINI_API_KEY = os.getenv(GEMINI_API_KEY_ENV_VAR)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", 'gemini-1.5-flash-latest') # Or other generative models
# Gemini Safety Settings (optional, adjust as needed)
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
# Gemini Generation Config (optional)
GEMINI_GENERATION_CONFIG = {
    "temperature": 0.7, # Controls randomness
    "top_p": 0.9,       # Nucleus sampling
    "top_k": 40,        # Top-k sampling
    # "max_output_tokens": 512, # Optional: limit output length
}

# --- Input Truncation (General Fallback) ---
# Simple token count limit as a fallback if specific model limits aren't handled
# (Approximation: 1 token ~= 4 chars in English)
MAX_LLM_INPUT_CHARS = 3500 # Adjust as needed (approx 800-900 tokens)