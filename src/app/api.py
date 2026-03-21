from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .models import QuestionRequest, QAResponse
from .services.qa_service import answer_question
from .services.indexing_service import index_pdf_file


app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.1.0",
)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Return explicit HTTP errors without converting them to generic 500s."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - simple demo handler
    """Catch-all handler for unexpected errors.

    FastAPI will still handle `HTTPException` instances and validation errors
    separately; this is only for truly unexpected failures so API consumers
    get a consistent 500 response body.
    """

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    """Submit a question about the vector databases paper.

    US-001 requirements:
    - Accept POST requests at `/qa` with JSON body containing a `question` field
    - Validate the request format and return 400 for invalid requests
    - Return 200 with `answer`, `draft_answer`, and `context` fields
    - Delegate to the multi-agent RAG service layer for processing
    """

    question = payload.question.strip()
    if not question:
        # Explicit validation beyond Pydantic's type checking to ensure
        # non-empty questions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    # Delegate to the service layer which runs the multi-agent QA graph.
    # Convert common provider/config/runtime failures into actionable API errors
    # instead of generic 500 responses.
    try:
        result = answer_question(question)
    except Exception as exc:
        exc_name = exc.__class__.__name__
        message = str(exc)

        lowered = message.lower()
        if (
            exc_name in {"RateLimitError", "ResourceExhausted"}
            or "insufficient_quota" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
            or "429" in lowered
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "LLM provider quota/rate limit exceeded. Update billing/quota for "
                    "the configured provider and retry."
                ),
            ) from exc

        if "not_found" in lowered or "model" in lowered and "not found" in lowered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Configured LLM model was not found. Check your model name "
                    "for the selected provider."
                ),
            ) from exc

        if exc_name in {"AuthenticationError", "PermissionDeniedError"}:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "LLM provider authentication failed. Check your API key "
                    "and environment configuration."
                ),
            ) from exc

        if isinstance(exc, RuntimeError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        raise

    return QAResponse(
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    """Upload a PDF and index it into the vector database.

    This endpoint:
    - Accepts a PDF file upload
    - Saves it to the local `data/uploads/` directory
    - Uses PyPDFLoader to load the document into LangChain `Document` objects
    - Indexes those documents into the configured Pinecone vector store
    """

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    contents = await file.read()
    file_path.write_bytes(contents)

    # Index the saved PDF
    chunks_indexed = index_pdf_file(file_path)

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
        "message": "PDF indexed successfully.",
    }
