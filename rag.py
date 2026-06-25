import os

from dotenv import load_dotenv

import streamlit as st

from ingestion import get_pinecone_client

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_classic.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_classic.chains.retrieval import (
    create_retrieval_chain
)

# =====================================================
# Environment Variables
# =====================================================

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =====================================================
# Cached Embeddings
# =====================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =====================================================
# Cached Vector Store
# =====================================================

@st.cache_resource(show_spinner=False)
def get_vectorstore():

    get_pinecone_client()

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embeddings()
    )

# =====================================================
# Cached Retriever
# =====================================================

@st.cache_resource(show_spinner=False)
def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

# =====================================================
# Cached LLM
# =====================================================

@st.cache_resource(show_spinner=False)
def get_llm():

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile"
    )

# =====================================================
# Cached RAG Chain
# =====================================================

@st.cache_resource(show_spinner=False)
def get_rag_chain():

    llm = get_llm()

    retriever = get_retriever()

    # -------------------------------------------
    # Contextualize Follow-up Questions
    # -------------------------------------------

    contextualize_q_prompt = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Given the chat history and the latest
                    user question, rewrite the question
                    as a standalone question.

                    Do NOT answer the question.

                    Only rewrite it if necessary.
                    """
                ),

                MessagesPlaceholder(
                    "chat_history"
                ),

                (
                    "human",
                    "{input}"
                )
            ]
        )
    )

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )
    )

    # -------------------------------------------
    # QA Prompt
    # -------------------------------------------

    qa_prompt = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful research assistant.

                    Use ONLY the information
                    present in the retrieved context.

                    If the answer is not found
                    in the documents, respond:

                    "I could not find that information
                    in the uploaded documents."

                    Context:

                    {context}
                    """
                ),

                MessagesPlaceholder(
                    "chat_history"
                ),

                (
                    "human",
                    "{input}"
                )
            ]
        )
    )

    # -------------------------------------------
    # Stuff Documents Chain
    # -------------------------------------------

    question_answer_chain = (
        create_stuff_documents_chain(
            llm,
            qa_prompt
        )
    )

    # -------------------------------------------
    # Retrieval Chain
    # -------------------------------------------

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain

# =====================================================
# Ask Question
# =====================================================

def ask_question(
    question,
    chat_history
):

    rag_chain = get_rag_chain()

    response = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

    return response["answer"]

