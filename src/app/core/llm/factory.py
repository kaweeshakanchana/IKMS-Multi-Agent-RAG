"""Factory functions for creating LangChain v1 LLM instances."""

from langchain_openai import ChatOpenAI

from ..config import get_settings


def create_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Create a LangChain v1 ChatOpenAI instance.

    Args:
        temperature: Model temperature (default: 0.0 for deterministic outputs).

    Returns:
        Configured ChatOpenAI instance.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "Missing OpenAI API key. Set `OPENAI_API_KEY` (or `openai_api_key`) "
            "in your environment or in a local `.env` file."
        )
    return ChatOpenAI(
        model=settings.openai_model_name,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
