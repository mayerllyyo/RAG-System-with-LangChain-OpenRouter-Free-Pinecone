"""
RAG Chain (Two-Step Approach) — Ecommerce FAQ Chatbot
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.llm import get_llm
from utils.vector_store import get_vector_store

print("Initialising vector store and LLM...")
vector_store = get_vector_store()
llm = get_llm()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

RAG_PROMPT_TEMPLATE = """You are a helpful and friendly e-commerce customer support assistant.
Answer the customer's question using ONLY the information provided in the FAQ context below.
If the context does not contain enough information, say:
"I'm sorry, I don't have that information. Please contact our support team directly."

FAQ Context:
{context}

Customer Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

def format_docs(docs) -> str:
    """Concatenate retrieved FAQ Q&A pairs into a readable context block."""
    return "\n\n---\n\n".join(
        f"FAQ #{doc.metadata.get('index', '?')}:\n{doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def ask_chain(question: str) -> str:
    """
    Run the two-step RAG chain for a customer question.
    """
    print(f"\n{'='*60}")
    print(f"👤 Customer: {question}")
    print(f"{'='*60}")

    # Show retrieved FAQs for transparency
    retrieved = retriever.invoke(question)
    print(f"\nTop {len(retrieved)} retrieved FAQs:")
    for i, doc in enumerate(retrieved, 1):
        faq_q = doc.metadata.get("question", "")
        print(f"  [{i}] {faq_q}")

    print("\n🤖 Generating answer...")
    answer = rag_chain.invoke(question)
    print(f"\nAnswer:\n{answer}")
    return answer

def main():
    questions = [
        "How do I track my order?",
        "What is your return policy?",
        "Do you offer international shipping?",
        "I forgot my password, how can I reset it?",
        "Are there any discounts available for new customers?",
    ]

    for q in questions:
        ask_chain(q)
        print()

if __name__ == "__main__":
    main()