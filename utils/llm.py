"""

Creates a ChatOpenAI instance pointed at OpenRouter.

OpenRouter is a free-tier AI gateway that exposes an OpenAI-compatible REST API
"""

import os
from langchain_openai import ChatOpenAI

def get_llm(model: str = "mistralai/mistral-7b-instruct", temperature: float = 0) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance configured to use OpenRouter.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please add it to your .env file."
        )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-OpenRouter-Organization": "org-1a2b3c4d5e6f7g8h9i0j"
        },
    )
    return llm