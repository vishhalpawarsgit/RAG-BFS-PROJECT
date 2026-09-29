# RAG-BFS-PROJECT
1. Project Overview

This project implements a domain-specific Retrieval-Augmented Generation (RAG) question-answering system for Banking & Financial Services (BFS).

The application loads PDF documents, extracts and chunks their text, generates vector embeddings, stores the embeddings in FAISS, retrieves the most relevant chunks for a user question, and sends the retrieved context to an LLM to produce a grounded answer.

The application provides a continuous command-line interface (CLI).

2. Domain Chosen

Banking & Financial Services (BFS)

The system is designed for questions involving banking financial performance, financial statements, banking operations, regulation, risk management, credit, and related BFS topics.

3. Dataset Description

The final dataset should contain multiple genuine public BFS PDFs in the data/ directory.

Recommended sources:

Credentials ** SO didnt upload 

During development, synthetic BFS PDFs were used for controlled testing. These synthetic documents are clearly identified as synthetic and must not be represented as real-world financial information.

See REAL_DATA_SOURCES.md for official source pages.

4. Architecture

User Question
     |
     v
Retriever
     |
     v
Sentence Transformer Embedding
     |
     v
FAISS Vector Search
     |
     v
Top-K Relevant Chunks
     |
     v
Retrieved Document Context
     |
     v
OpenAI LLM
     |
     v
Grounded Answer + Sources

Ingestion pipeline

PDF files
   |
   v
PyMuPDF text extraction
   |
   v
Page-aware chunking
   |
   v
SentenceTransformer embeddings
   |
   v
FAISS index
   |
   v
Saved chunks + metadata
