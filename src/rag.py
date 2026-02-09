"""
Retrieval-Augmented Generation (RAG) pipeline.

This module implements a standard RAG pipeline for climate documentation:
1. Load and index documents
2. Retrieve relevant chunks
3. Generate answers with citations
"""

import os
import json
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .llm import generate_from_messages

# Default configuration
DEFAULT_CORPUS_PATH = "./data/climate_corpus.jsonl"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

# Global state (lazy loaded)
_vectorstore = None
_retriever = None
_embeddings = None


def load_corpus(path: Optional[str] = None) -> List[Document]:
    """
    Load a JSONL corpus where each line is a retrieval chunk.

    Required fields per line:
      - doc_id: unique chunk id
      - text: chunk content

    Any other fields will be stored as Document.metadata.

    Args:
        path: Path to the JSONL corpus file.

    Returns:
        List of LangChain Document objects.
    """
    if path is None:
        path = os.getenv("CORPUS_PATH", DEFAULT_CORPUS_PATH)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}")

    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "doc_id" not in obj or "text" not in obj:
                raise ValueError(
                    f"Invalid JSONL at line {line_no}: missing doc_id or text."
                )

            text = obj["text"]
            metadata = {k: v for k, v in obj.items() if k != "text"}
            docs.append(Document(page_content=text, metadata=metadata))

    if not docs:
        raise ValueError("Corpus loaded 0 documents. Check your file path/content.")

    print(f"Loaded {len(docs)} documents from corpus.")
    return docs


def build_vectorstore(
    docs: Optional[List[Document]] = None,
    corpus_path: Optional[str] = None,
    embed_model_name: Optional[str] = None,
) -> FAISS:
    """
    Build a FAISS vector store from documents.

    Args:
        docs: List of documents. If None, loads from corpus_path.
        corpus_path: Path to corpus file (used if docs is None).
        embed_model_name: Name of the embedding model.

    Returns:
        FAISS vector store.
    """
    global _vectorstore, _retriever, _embeddings

    if embed_model_name is None:
        embed_model_name = os.getenv("EMBED_MODEL_NAME", DEFAULT_EMBED_MODEL)

    if docs is None:
        docs = load_corpus(corpus_path)

    _embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
    _vectorstore = FAISS.from_documents(docs, _embeddings)

    print("Vector store built successfully.")
    return _vectorstore


def get_retriever(k: int = DEFAULT_TOP_K):
    """
    Get or create the retriever.

    Args:
        k: Number of documents to retrieve.

    Returns:
        FAISS retriever.
    """
    global _vectorstore, _retriever

    if _vectorstore is None:
        build_vectorstore()

    _retriever = _vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query: The search query.
        k: Number of documents to retrieve.

    Returns:
        List of retrieved documents.
    """
    retriever = get_retriever(k)
    return retriever.invoke(query)


def show_retrieval_evidence(
    query: str, retrieved_docs: List[Document], max_chars: int = 300
) -> None:
    """
    Display retrieved evidence (required for grading).

    Args:
        query: The original query.
        retrieved_docs: List of retrieved documents.
        max_chars: Maximum characters to show in snippet.
    """
    print("=" * 90)
    print("RAG EVIDENCE (REQUIRED)")
    print("=" * 90)
    print(f"Query: {query}")
    print(f"Retrieved: {len(retrieved_docs)} chunks\n")

    for i, d in enumerate(retrieved_docs, 1):
        doc_id = d.metadata.get("doc_id", "N/A")
        dtype = d.metadata.get("type", "N/A")
        var = d.metadata.get("var", "")
        title = d.metadata.get("title", "")
        snippet = d.page_content[:max_chars].strip() + (
            "..." if len(d.page_content) > max_chars else ""
        )

        print(f"[{i}] doc_id={doc_id} | type={dtype} | var={var} | title={title}")
        print(f"Snippet: {snippet}\n")


def build_rag_messages(
    context_docs: List[Document], question: str
) -> List[Dict[str, str]]:
    """
    Build RAG prompt with document context and citations.

    Args:
        context_docs: Retrieved documents.
        question: User's question.

    Returns:
        List of messages for the LLM.
    """
    ctx_blocks = []
    for d in context_docs:
        doc_id = d.metadata.get("doc_id", "unknown")
        ctx_blocks.append(f"[{doc_id}]\n{d.page_content.strip()}")

    context = "\n\n".join(ctx_blocks)

    system = (
        "You are a retrieval-augmented assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        'If the context is insufficient, say: "I don\'t have enough information from the provided documents."\n'
        "You MUST cite sources using doc_id in square brackets, e.g., [var_t2m], [dataset_overview].\n"
        "Do NOT cite anything that is not in the context."
    )

    user = f"Context:\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rag_answer(
    question: str, k: int = DEFAULT_TOP_K, show_evidence: bool = True
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline.

    1. Retrieve top-k chunks
    2. Display evidence (if show_evidence=True)
    3. Build grounded prompt with [doc_id] headers
    4. Generate answer with citations

    Args:
        question: User's question.
        k: Number of documents to retrieve.
        show_evidence: Whether to print retrieved evidence.

    Returns:
        Dictionary with 'answer' and 'evidence' keys.
    """
    # Retrieve
    retrieved_docs = retrieve(question, k)

    if show_evidence:
        show_retrieval_evidence(question, retrieved_docs)

    # Build messages
    messages = build_rag_messages(retrieved_docs, question)

    # Generate
    answer = generate_from_messages(messages)

    # Format evidence
    evidence = []
    for d in retrieved_docs:
        evidence.append(
            {
                "doc_id": d.metadata.get("doc_id", "N/A"),
                "title": d.metadata.get("title", "N/A"),
                "var": d.metadata.get("var", ""),
                "text": d.page_content,
            }
        )

    return {"answer": answer, "evidence": evidence}
