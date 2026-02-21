"""
Custom embeddings wrapper that uses OpenRouter-compatible API endpoint
with the OpenAI embeddings interface.
"""

import os
from langchain_openai import OpenAIEmbeddings


DEFAULT_EMBEDDINGS_MODEL = "openai/text-embedding-3-small"
EMBEDDINGS_DIMENSIONS = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}


def get_embedding_dimension(model: str | None = None) -> int:
    """
    Return the expected vector dimension for the embeddings model.
    """
    model_name = model or os.environ.get("OPENROUTER_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL)
    override = os.environ.get("OPENROUTER_EMBEDDINGS_DIMENSION")
    if override:
        return int(override)
    if model_name not in EMBEDDINGS_DIMENSIONS:
        raise ValueError(
            "Unknown embeddings model dimension. "
            "Set OPENROUTER_EMBEDDINGS_DIMENSION in .env to the correct size."
        )
    return EMBEDDINGS_DIMENSIONS[model_name]


def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns an OpenAIEmbeddings instance configured to use OpenRouter.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    model = os.environ.get("OPENROUTER_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL)
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return embeddings