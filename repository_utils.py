import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    SUMMARY_MAPPING_PATH
)

class ConcertRAGRepository:
    """Manages the storage and retrieval of concert tour document summaries."""

    def __init__(self):
        """Initializes the repository, loading existing data or creating new."""
        print("Initializing RAG Repository...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.summary_map = {} # Maps FAISS index ID -> summary text
        self.next_id = 0
        self._load_repository()
        print(f"Repository initialized. Index size: {self.index.ntotal if self.index else 0}")

    def _load_repository(self):
        """Loads the FAISS index and summary mapping from disk if they exist."""
        loaded_index = False
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(FAISS_INDEX_PATH)
                print(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
                loaded_index = True
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Creating a new one.")
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

        else:
            print("FAISS index file not found. Creating a new one.")
            # Using IndexIDMap to map our sequential IDs to FAISS internal IDs
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
            # IndexFlatL2 performs simple Euclidean distance search

        if os.path.exists(SUMMARY_MAPPING_PATH):
            try:
                with open(SUMMARY_MAPPING_PATH, 'r', encoding='utf-8') as f:
                    # Load keys as integers
                    self.summary_map = {int(k): v for k, v in json.load(f).items()}
                # Ensure next_id is correctly set based on loaded data
                if self.summary_map:
                    self.next_id = max(self.summary_map.keys()) + 1
                print(f"Loaded summary map from {SUMMARY_MAPPING_PATH}. Count: {len(self.summary_map)}")

                # Consistency check: index size vs map size
                if loaded_index and self.index.ntotal != len(self.summary_map):
                     print(f"Warning: Index size ({self.index.ntotal}) differs from map size ({len(self.summary_map)}). Check data consistency.")
                     # Potentially add logic here to rebuild index or prune map if critical

            except Exception as e:
                print(f"Error loading summary map: {e}. Initializing empty map.")
                self.summary_map = {}
                self.next_id = 0
                # If index was loaded but map failed, maybe clear the index too?
                # self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
                # print("Cleared FAISS index due to map loading failure.")

        else:
            print("Summary map file not found. Initializing empty map.")
            self.summary_map = {}
            self.next_id = 0

    def add_document_summary(self, summary: str) -> int:
        """
        Adds a document summary to the repository.

        Args:
            summary: The text summary to add.

        Returns:
            The unique ID assigned to this summary.
        """
        if not summary:
            print("Warning: Attempted to add an empty summary.")
            return -1

        try:
            # Generate embedding
            embedding = self.model.encode([summary], convert_to_numpy=True)

            # Add to FAISS index with our unique ID
            current_id = self.next_id
            ids_to_add = np.array([current_id]).astype('int64')
            self.index.add_with_ids(embedding, ids_to_add)

            # Store the summary text mapped to the ID
            self.summary_map[current_id] = summary
            self.next_id += 1

            # Persist changes immediately (can be optimized for bulk adds)
            self._save_repository()
            print(f"Added summary with ID {current_id}. Index size: {self.index.ntotal}")
            return current_id

        except Exception as e:
            print(f"Error adding document summary: {e}")
            # Consider rollback or cleanup if needed
            return -1

    def search_relevant_summaries(self, query: str, k: int = 3) -> list[str]:
        """
        Searches for summaries relevant to the query.

        Args:
            query: The user's question or search term.
            k: The number of top relevant summaries to retrieve.

        Returns:
            A list of the most relevant summary strings.
        """
        if not query or self.index.ntotal == 0:
            return [] # No query or nothing in the index

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)

            # Search the FAISS index
            # D: distances, I: indices (our IDs in this case)
            distances, indices = self.index.search(query_embedding, k=min(k, self.index.ntotal))

            # Retrieve the corresponding summaries using the indices (IDs)
            relevant_summaries = []
            if indices.size > 0:
                for idx in indices[0]: # indices is shape (1, k)
                    if idx != -1: # FAISS returns -1 for non-existent neighbors
                       summary = self.summary_map.get(idx)
                       if summary:
                           relevant_summaries.append(summary)
                       else:
                           print(f"Warning: ID {idx} found in index but not in summary map.")

            return relevant_summaries

        except Exception as e:
            print(f"Error searching for relevant summaries: {e}")
            return []

    def _save_repository(self):
        """Saves the FAISS index and summary mapping to disk."""
        try:
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            # Convert keys to strings for JSON compatibility
            summary_map_str_keys = {str(k): v for k, v in self.summary_map.items()}
            with open(SUMMARY_MAPPING_PATH, 'w', encoding='utf-8') as f:
                json.dump(summary_map_str_keys, f, indent=4, ensure_ascii=False)
            # print("Repository state saved.") # Optional: can be noisy
        except Exception as e:
            print(f"Error saving repository state: {e}")

    def get_total_documents(self) -> int:
        """Returns the total number of documents stored."""
        return self.index.ntotal if self.index else 0

# --- Repository Singleton ---
# Create a single instance to be used throughout the application
# This avoids reloading models and data repeatedly
_repository_instance = None

def get_repository() -> ConcertRAGRepository:
    """Gets the singleton instance of the ConcertRAGRepository."""
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = ConcertRAGRepository()
    return _repository_instance