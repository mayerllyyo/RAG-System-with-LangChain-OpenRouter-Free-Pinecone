"""
Handles creation and retrieval of a Pinecone vector store.

Pinecone is a managed vector database. We use it to persist document
embeddings so they survive across sessions without re-indexing.
"""

import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from utils.embeddings import get_embedding_dimension, get_embeddings


def _get_index_dimension(pc: Pinecone, index_name: str) -> int | None:
    """
    Return the dimension for an existing Pinecone index if available.
    """
    try:
        desc = pc.describe_index(index_name)
    except Exception:
        return None
    if hasattr(desc, "dimension"):
        return desc.dimension
    if isinstance(desc, dict):
        return desc.get("dimension")
    return None

def get_or_create_index(index_name: str, dimension: int = 1536) -> str:
    """
    Creates a Pinecone serverless index if it does not already exist.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"[Pinecone] Creating index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[Pinecone] Index '{index_name}' created.")
        return index_name

    existing_dim = _get_index_dimension(pc, index_name)
    if existing_dim is not None and existing_dim != dimension:
        fallback_name = f"{index_name}-d{dimension}"
        print(
            f"[Pinecone] Index '{index_name}' has dimension {existing_dim}, "
            f"expected {dimension}. Using '{fallback_name}' instead."
        )
        if fallback_name not in existing_indexes:
            print(f"[Pinecone] Creating index '{fallback_name}' ...")
            pc.create_index(
                name=fallback_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"[Pinecone] Index '{fallback_name}' created.")
        return fallback_name

    print(f"[Pinecone] Index '{index_name}' already exists.")
    return index_name


def get_vector_store(index_name: str | None = None) -> PineconeVectorStore:
    """
    Returns a PineconeVectorStore connected to the specified index.
    """
    if index_name is None:
        index_name = os.environ.get("PINECONE_INDEX_NAME", "rag-langchain-index")

    embeddings = get_embeddings()
    index_dimension = get_embedding_dimension(embeddings.model)
    index_name = get_or_create_index(index_name, dimension=index_dimension)

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    return vector_store