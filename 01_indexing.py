"""
RAG Indexing Pipeline — Ecommerce FAQ Dataset
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from utils.data_loader import load_faq_documents
from utils.vector_store import get_vector_store

DATASET_PATH = Path("data/Ecommerce_FAQ_Chatbot_dataset.json")

def load_documents():
    print("\n[1/2] Loading FAQ documents from JSON dataset...")
    docs = load_faq_documents(DATASET_PATH)

    # Preview the first two documents
    print("\n  Preview of first 2 documents:")
    for doc in docs[:2]:
        print(f"  {doc.page_content[:140]}...")
        print(f"  metadata: {doc.metadata}")

    return docs

def store_documents(docs):
    print(f"\n[2/2] Embedding {len(docs)} FAQ pairs and storing in Pinecone...")
    vector_store = get_vector_store()
    ids = vector_store.add_documents(documents=docs)
    print(f"Stored {len(ids)} document vectors.")
    print(f"Sample IDs: {ids[:3]}")
    return ids

def main():
    print("RAG Indexing Pipeline")
    print("Dataset: Ecommerce FAQ Chatbot (79 Q&A pairs)")

    docs = load_documents()
    store_documents(docs)

if __name__ == "__main__":
    main()
    