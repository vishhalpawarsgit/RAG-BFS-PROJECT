import os
import json
import logging
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
STORE_DIR = BASE_DIR / "store"

VECTOR_DIR = STORE_DIR / "vector_index"
CHUNKS_DIR = STORE_DIR / "chunks"

VECTOR_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PDF Loading
# ---------------------------------------------------------

def load_pdfs():
    """
    Load all PDF files from the data directory.

    Returns:
        List of dictionaries containing text and metadata.
    """

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:

        logger.info("Loading PDF: %s", pdf_path.name)

        try:
            pdf = fitz.open(pdf_path)

            for page_number, page in enumerate(pdf, start=1):

                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    {
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number
                    }
                )

            pdf.close()

        except Exception as exc:
            logger.error(
                "Failed to process %s: %s",
                pdf_path.name,
                exc
            )

    logger.info(
        "Loaded %d pages from %d PDF files",
        len(documents),
        len(pdf_files)
    )

    return documents


# ---------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------

def create_chunks(documents):
    """
    Split document text into overlapping chunks.

    Each chunk keeps its source PDF and page number.
    """

    chunks = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": document["source"],
                        "page": document["page"]
                    }
                )

            next_start = end - CHUNK_OVERLAP

            if next_start <= start:
                break

            start = next_start

    logger.info(
        "Created %d chunks",
        len(chunks)
    )

    return chunks


# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------

def generate_embeddings(chunks):
    """
    Generate vector embeddings for all chunks.
    """

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL
    )

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    logger.info(
        "Generating embeddings for %d chunks...",
        len(texts)
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    return embeddings


# ---------------------------------------------------------
# Save FAISS Index
# ---------------------------------------------------------

def save_vector_index(embeddings):
    """
    Store embeddings inside a FAISS index.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    index_path = VECTOR_DIR / "faiss.index"

    faiss.write_index(
        index,
        str(index_path)
    )

    logger.info(
        "FAISS index saved to: %s",
        index_path
    )

    return index


# ---------------------------------------------------------
# Save Chunk Metadata
# ---------------------------------------------------------

def save_chunks(chunks):
    """
    Save chunks and their metadata as JSON.
    """

    chunks_path = CHUNKS_DIR / "chunks.json"

    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    logger.info(
        "Chunk metadata saved to: %s",
        chunks_path
    )


# ---------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------

def main():

    logger.info("Starting BFS RAG ingestion pipeline...")

    documents = load_pdfs()

    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the PDFs."
        )

    embeddings = generate_embeddings(chunks)

    save_vector_index(embeddings)

    save_chunks(chunks)

    logger.info(
        "Ingestion completed successfully."
    )

    logger.info(
        "Documents/pages processed: %d",
        len(documents)
    )

    logger.info(
        "Chunks created: %d",
        len(chunks)
    )

    logger.info(
        "Embedding dimension: %d",
        embeddings.shape[1]
    )


if __name__ == "__main__":
    main()