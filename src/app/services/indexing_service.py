"""Service functions for indexing documents into the vector database."""

from pathlib import Path



from ..core.retrieval.vector_store import index_documents


def index_pdf_file(file_path: Path) -> dict:
    """Load a PDF from disk and index it into the vector DB.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        Metadata about the indexing operation.
    """

    return index_documents(file_path)
