"""Vector store wrapper for Pinecone integration with LangChain."""

from pathlib import Path
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create a PineconeVectorStore instance configured from settings."""
    settings = get_settings()

    if not settings.pinecone_api_key or not settings.pinecone_index_name:
        raise RuntimeError(
            "Missing Pinecone configuration. Set `PINECONE_API_KEY` "
            "and `PINECONE_INDEX_NAME` in your environment or `.env`."
        )

    # Import lazily to avoid crashing app startup if the dependency isn't installed yet.
    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model_name,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

def get_retriever(k: int | None = None):
    """Get a Pinecone retriever instance.

    Args:
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        PineconeVectorStore instance configured as a retriever.
    """
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone for a given query.

    Args:
        query: Search query string.
        k: Number of documents to retrieve (defaults to config value).

    Returns:
        List of Document objects with metadata (including page numbers).
    """
    retriever = get_retriever(k=k)
    return retriever.invoke(query)

def index_documents(file_path: Path) -> dict:
    """Index a PDF file into the Pinecone vector store and verify results.

    Args:
        file_path: PDF file path to load and split into documents.

    Returns:
        Dictionary containing counts from the split and Pinecone stats.
    """

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    if not docs:
        raise ValueError(f"No pages extracted from PDF: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    vector_store = _get_vector_store()

    # Capture index stats before and after upsert for correctness validation.
    before_stats = vector_store.index.describe_index_stats()
    before_count = before_stats.get("total_vector_count", 0)

    upsert_ids = vector_store.add_documents(texts)

    after_stats = vector_store.index.describe_index_stats()
    after_count = after_stats.get("total_vector_count", 0)

    return {
        "chunks_indexed": len(texts),
        "upserted_count": len(upsert_ids),
        "index_count_before": before_count,
        "index_count_after": after_count,
    }