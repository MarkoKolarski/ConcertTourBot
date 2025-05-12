# 🎶 Concert Tour Bot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-red?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Enabled-green)](https://faiss.ai/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Integrated-orange?logo=google)](https://ai.google/)

## Overview

**Concert Tour Bot** is a Python-based service developed to intelligently manage and retrieve domain-specific information about upcoming concert tours for the 2025–2026 season. The service uses **Retrieval-Augmented Generation (RAG)** for document storage and querying, ensuring that answers are **strictly grounded in user-provided documents** rather than general knowledge.

This bot supports:

### 🎯 Core Functionalities (available in all environments)
- Adding new concert tour domain related documents (e.g., schedules, venues, artists, logistics) via document ingestion.
- Validating domain relevance and summarizing relevant documents.
- Answering user queries strictly based on ingested and indexed documents using a RAG (Retrieval-Augmented Generation) system.

### ⭐ Bonus Features *(available only in the Streamlit UI interface)*
- **Streamlit UI** for seamless interaction with the bot via a web interface.
- Performing **online searches** for concert details when no documents are provided, extracting tour info from the web (e.g., SerpAPI/Bing API).


---


## ✨ Features



### 1. Document Management
- Accepts plain-text documents related to concert tours.
- Validates if the document belongs to the concert tour domain.
- Summarizes relevant documents using Large Language Models (LLMs).
- Stores document summaries in a **FAISS vector database** for fast retrieval.

### 2. Query Answering
- Answers user questions using retrieved documents and LLMs.
- Ensures responses are **strictly based** on ingested content.



#### 3. Streamlit Web Interface
- Chat-based interface for commands and questions.
- Sidebar for managing settings and API keys.
- Real-time updates on repository size and indexing status.

#### 4. Optional: Online Search *(Streamlit UI only)*
- This feature is **only available through the Streamlit web interface**.
- If no relevant data is found in the local RAG system, the bot can perform live web searches via **SerpAPI**.
- Synthesizes search results into concise, LLM-generated answers.
- Requires a valid `SERP_API_KEY`.



---

## 🧠 Design Choices

### Retrieval-Augmented Generation (RAG)
- Uses **FAISS** for storing vectorized summaries.
- **SentenceTransformers** generate embeddings for similarity search.

### LLM Integration
- **Google Gemini**: Cloud-based LLM for enhanced capabilities.
- **Hugging Face**: Local models for summarization and QA.
- Users can switch between providers via the UI.

---

## 🗂️ File Structure

```
ConcertTourBot/
│
├── main.py                   # CLI interface
├── app_streamlit.py          # Streamlit web interface
│
├── repository_utils.py       # FAISS index management
├── document_processor.py     # Document validation & summarization
├── qa_handler.py             # Question answering logic
├── llm_integrator.py         # LLM provider integration
│
├── config.py                 # Configuration variables
├── .env                      # API keys (not included in repo)
│
├── Dockerfile                # Docker configuration for CLI app
├── Dockerfile.streamlit      # Docker configuration for Streamlit app
│
├── .streamlit/
│   ├── style.css             # Custom styles for Streamlit
│   └──config.toml               # Streamlit configuration
│
├── concert_tour_index.faiss  # FAISS index (auto-generated)
├── summary_mapping.json      # Summary metadata (auto-generated)
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Installation & Setup

You can run this project in **two ways**: via **classic Python setup** or using **Docker containers**.

---

## 📦 Option 1: Classic Python Installation

### ✅ Prerequisites

- Python **3.10+**
- `pip` (Python package manager)
- `git`

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/MarkoKolarski/ConcertTourBot.git
cd ConcertTourBot
```

---

### 🧱 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔐 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_google_gemini_api_key

# SerpAPI Key for online concert search
SERPAPI_KEY=your_serpapi_key
```

---

### 🚀 5. Run the Application

⚠️ **Before you begin:**  
You must select your preferred LLM backend.

 - 🌐 **Google Gemini API** (API) (Recommended):  
   - Requires a valid `GOOGLE_API_KEY`.  
   - Delivers highly accurate answers grounded in ingested content.  
   - Best suited for production-quality responses.

 - 🧠 **Local Hugging Face Model** (Local):  
   - Lightweight and fast.  
   - Suitable for **basic queries only**.  
   - May **hallucinate** and is **not recommended** for precise answers.



 🔍 Additionally, if you wish to enable **live web search** (bonus feature), you must provide a valid `SERP_API_KEY`.

---

#### 🖥️ CLI Mode

```bash
python main.py
```

**Available Commands:**

- `ADD: <text>` – Add a concert document
- `QUERY: <question>` – Ask a question
- `COUNT` – View document count
- `PROVIDER` – View current LLM provider
- `CHANGE_PROVIDER` – Switch LLM provider
- `HELP` – Show available commands
- `EXIT` – Exit the program

---

#### 🌐 Streamlit Mode

```bash
streamlit run app_streamlit.py
```

Open your browser and visit: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Option 2: Docker Installation

### ✅ Prerequisites

- [Docker installed](https://www.docker.com/get-started) and running

---

### 🏗 1. Build Docker Images

```bash
# CLI version
docker build -t concert-bot .

# Streamlit version
docker build -t concert-bot-streamlit -f Dockerfile.streamlit .
```

---

### 🚀 2. Run Containers

#### ▶ CLI Mode

```bash
docker run --env-file .env -it concert-bot
```

---

#### ▶ Streamlit Mode

```bash
docker run --env-file .env -p 8501:8501 concert-bot-streamlit
```

Then go to: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Example Usage

You can interact with the bot through both the **CLI** and the **Streamlit UI**. The behavior is identical in both environments — with the exception of **online search**, which is only available via **Streamlit**.

---

### 📄 Add Document

**Command:**
```bash
ADD: Taylor Swift will perform in Paris, Berlin, and Rome during July 2025 as part of her Eras Tour extension.
```

**Output:**
```
--- Ingestion Confirmation ---
Document successfully added.
LLM (GEMINI) Summary:
'Taylor Swift's Eras Tour will extend to Paris, Berlin, and Rome in July 2025.'
-----------------------------
```

---

### ❓ Ask a Question

**Command:**
```bash
QUERY: Where is Taylor Swift performing in July 2025?
```

**Output:**
```
--- Answer ---
Paris, Berlin, and Rome.
--------------
```



---

### 🌐 Online Search *(Streamlit Only)*

If no relevant documents exist in your local index, the bot can attempt to answer the query by performing a **live web search** (if a valid `SERP_API_KEY` is provided).

**Example Query (no prior documents added):**
```bash
Kendrick Lamar
```

**Output (Streamlit UI with Google Gemini + SerpAPI):**
```
Information about Kendrick Lamar's upcoming concerts:

Kendrick Lamar's 2025 tour includes dates in April and June. April shows are in Minneapolis, MN (April 19th), Houston, TX (April 23rd), Arlington, TX (April 26th), and Atlanta, GA (April 29th). A May 3rd show in Charlotte, NC is also listed. A June 6th, 2025 concert at Soldier Field is mentioned, with a total of 37 concerts across 10 unspecified locations planned for the tour. The Arlington, TX show on April 26th is part of a "Grand National Tour" with SZA.

```

> 💡 *All responses in this section were generated using the Google Gemini LLM backend.*


## 🧰 Troubleshooting

| Issue                         | Solution |
|------------------------------|----------|
| 🔑 Missing API Keys           | Ensure `.env` file is correctly set |
| 📦 FAISS Index Errors         | Delete `concert_tour_index.faiss` and `summary_mapping.json` to reset |
| 🌐 Streamlit Fails to Launch  | Install all dependencies: `pip install -r requirements.txt` |
| 🐳 Docker Issues              | Verify Docker is running, check logs with `docker logs <container_id>` |

---