"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for the /qa endpoint."""

    question: str = Field(
        ...,
        description="User's question about the knowledge base.",
        min_length=1,
    )


class QAResponse(BaseModel):
    """Response model for the /qa endpoint."""

    answer: str = Field(
        ...,
        description="The answer to the user's question.",
    )
    context: str = Field(
        default="",
        description="Retrieved context used to generate the answer.",
    )
