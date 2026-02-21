import os
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_core.messages import HumanMessage
try:
    from langchain.agents import create_agent
    _CREATE_AGENT_USES_SYSTEM_PROMPT = True
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent
    _CREATE_AGENT_USES_SYSTEM_PROMPT = False

from utils.llm import get_llm
from utils.vector_store import get_vector_store

print("Initialising vector store and LLM...")
vector_store = get_vector_store()
llm = get_llm(model=os.getenv("OPENROUTER_TOOL_MODEL", "openai/gpt-4o-mini"))

@tool(response_format="content_and_artifact")
def retrieve_faq_context(query: str):
    """
    Search the e-commerce FAQ knowledge base for answers relevant to the query.

    Use this tool to look up information about:
    - Accounts, login, registration
    - Payments, discounts, promo codes
    - Orders, tracking, cancellations
    - Shipping, delivery times, international orders
    - Returns, refunds, exchanges
    - Products, availability, reviews
    - Customer support contact
    """
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"[FAQ #{doc.metadata.get('index', '?')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


tools = [retrieve_faq_context]

system_prompt = (
    "You are a friendly and professional e-commerce customer support assistant. "
    "Your knowledge comes exclusively from the company FAQ database. "
    "Always use the retrieve_faq_context tool to search for answers before responding. "
    "If the FAQ does not contain enough information, politely say so and suggest "
    "contacting customer support directly. "
    "Keep answers clear, concise, and helpful."
)

if _CREATE_AGENT_USES_SYSTEM_PROMPT:
    agent = create_agent(llm, tools, system_prompt=system_prompt)
else:
    agent = create_agent(llm, tools, prompt=system_prompt)


def ask_agent(question: str) -> str:
    """
    Send a customer question to the RAG agent and return the final answer.
    """
    print(f"\n{'='*60}")
    print(f"👤 Customer: {question}")
    print(f"{'='*60}")

    final_answer = ""
    for event in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        last_msg.pretty_print()
        if hasattr(last_msg, "content") and last_msg.type == "ai":
            final_answer = last_msg.content

    return final_answer

def main():
    questions = [
        # Simple single-topic question
        "How do I create an account?",

        # Multi-topic question (agent may call the tool twice)
        "What payment methods do you accept, and can I use a promo code at checkout?",

        # Question requiring chained retrieval
        (
            "I want to return a product I received damaged. "
            "What is your return policy and how long will the refund take?"
        ),
    ]

    for q in questions:
        ask_agent(q)
        print()

if __name__ == "__main__":
    main()