"""
Custom document loader for the Ecommerce FAQ Chatbot Dataset.
"""

import json
import os
from pathlib import Path
from langchain_core.documents import Document


def load_faq_documents(json_path: str | Path) -> list[Document]:
    """
    Load the FAQ JSON dataset and convert each Q&A pair into a Document.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{json_path}'.\n"
            "Please download it from Kaggle and place it in the data/ folder:\n"
            "  https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset\n"
            "  data/Ecommerce_FAQ_Chatbot_dataset.json"
        )

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    questions = raw.get("questions", [])
    if not questions:
        raise KeyError("Expected key 'questions' not found in dataset JSON.")

    documents = []
    for i, item in enumerate(questions):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()

        content = f"Q: {q}\nA: {a}"

        doc = Document(
            page_content=content,
            metadata={
                "source": "Ecommerce_FAQ_Chatbot_dataset",
                "question": q,
                "index": i,
            },
        )
        documents.append(doc)

    print(f"[DataLoader] Loaded {len(documents)} FAQ documents from '{json_path.name}'.")
    return documents