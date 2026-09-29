import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_INDEX_PATH = (
    BASE_DIR
    / "store"
    / "vector_index"
    / "faiss.index"
)

CHUNKS_PATH = (
    BASE_DIR
    / "store"
    / "chunks"
    / "chunks.json"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Retriever Class
# ---------------------------------------------------------

class Retriever:
    """
    Retrieves the most relevant document chunks
    from the FAISS vector database.
    """

    def __init__(
        self,
        index_path=VECTOR_INDEX_PATH,
        chunks_path=CHUNKS_PATH,
        top_k=DEFAULT_TOP_K
    ):

        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.top_k = top_k

        self._validate_files()

        logger.info("Loading FAISS index...")

        self.index = faiss.read_index(
            str(self.index_path)
        )

        logger.info(
            "FAISS index loaded. Total vectors: %d",
            self.index.ntotal
        )

        logger.info(
            "Loading chunk metadata..."
        )

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        logger.info(
            "Loaded %d chunks.",
            len(self.chunks)
        )

        logger.info(
            "Loading embedding model: %s",
            EMBEDDING_MODEL
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    # -----------------------------------------------------
    # Validate Required Files
    # -----------------------------------------------------

    def _validate_files(self):

        if not self.index_path.exists():

            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}. "
                "Run ingest.py first."
            )

        if not self.chunks_path.exists():

            raise FileNotFoundError(
                f"Chunk metadata not found: {self.chunks_path}. "
                "Run ingest.py first."
            )

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=None
    ):
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of dictionaries containing:
            - text
            - source
            - page
            - score
        """

        if not query or not query.strip():

            return []

        query = query.strip()

        if top_k is None:
            top_k = self.top_k

        # Do not request more results than available chunks.
        top_k = min(
            top_k,
            len(self.chunks)
        )

        # ---------------------------------------------
        # Create query embedding
        # ---------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        # ---------------------------------------------
        # Search FAISS
        # ---------------------------------------------

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # ---------------------------------------------
        # Build results
        # ---------------------------------------------

        for distance, index_position in zip(
            distances[0],
            indices[0]
        ):

            # FAISS may return -1 if no result exists.
            if index_position < 0:
                continue

            chunk = self.chunks[index_position]

            result = {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(distance)
            }

            results.append(result)

        logger.info(
            "Retrieved %d chunks for query: %s",
            len(results),
            query
        )

        return results


# ---------------------------------------------------------
# Standalone Testing
# ---------------------------------------------------------

def main():

    print("\nBFS RAG Retriever Test")
    print("=" * 50)

    retriever = Retriever()

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit'): "
        ).strip()

        if query.lower() in {
            "exit",
            "quit"
        }:
            print("Exiting retriever test.")
            break

        if not query:

            print(
                "Please enter a question."
            )

            continue

        results = retriever.retrieve(
            query,
            top_k=3
        )

        print("\nRetrieved Results")
        print("-" * 50)

        if not results:

            print(
                "No relevant results found."
            )

            continue

        for number, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {number}"
            )

            print(
                f"Source: {result['source']}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Distance: {result['score']:.4f}"
            )

            print(
                f"Text:\n{result['text']}"
            )


if __name__ == "__main__":
    main()