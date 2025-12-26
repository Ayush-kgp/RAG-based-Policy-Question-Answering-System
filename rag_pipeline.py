from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from typing import List
import os


# Prompt v1 (Initial)
def build_prompt_v1(context: str, question: str) -> str:
    """
    Prompt v1: Basic context-based QA.
    No strict hallucination control or fallback.
    """

    return f"""
Answer the question using the following context.

Context:
{context}

Question:
{question}

Answer:
"""

# Prompt v2 (Improved)
def build_prompt_v2(context: str, question: str) -> str:
    """
    Prompt v2: Strict grounding + graceful fallback
    """

    return f"""
You are a question-answering assistant for company policy documents.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not present in the context, say:
   "The information is not available in the provided documents."
3. Do NOT make assumptions or add external knowledge.
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""


# Retrieval
def retrieve_context(
    vectorstore,
    question: str,
    k: int = 8
) -> List[Document]:
    """
    Retrieve top-k relevant chunks from the vector store.
    """

    return vectorstore.similarity_search(question, k=k)


# RAG Pipeline
def answer_question(
    vectorstore,
    question: str,
    k: int = 8,
    prompt_version: str = "v2"
) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build grounded prompt (v1 or v2)
    3. Query LLM
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")

    retrieved_docs = retrieve_context(vectorstore, question, k=k)

    if not retrieved_docs:
        return "The information is not available in the provided documents."

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('doc_name')} | {doc.metadata.get('section')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    # Select prompt
    if prompt_version == "v1":
        prompt = build_prompt_v1(context, question)
    else:
        prompt = build_prompt_v2(context, question)

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content.strip()
