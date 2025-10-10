# api/langchain_utils.py
import os
import traceback
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional

# We'll lazy-init retriever to avoid heavy work at import time
_retriever = None

# prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_retriever():
    """
    Lazily import the vectorstore retriever so imports won't block server start.
    """
    global _retriever
    if _retriever is None:
        try:
            from api.chroma_utils import vectorstore
            _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print("Failed to initialize retriever:", e)
            traceback.print_exc()
            raise
    return _retriever

def get_rag_chain(model: str = "gemini-2.5-pro"):
    """
    Build a RAG chain on demand. Keep LLM creation here (so it uses env vars when invoked).
    """
    try:
        llm = ChatGoogleGenerativeAI(model=model)
    except Exception as e:
        # Surface a clear error if LLM creation fails (e.g. missing credentials)
        print("Failed to initialize ChatGoogleGenerativeAI:", e)
        traceback.print_exc()
        raise

    retriever = get_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
