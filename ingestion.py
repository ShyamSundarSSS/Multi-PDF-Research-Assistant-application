import os
import json
import hashlib
import time

from dotenv import load_dotenv

import streamlit as st

from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore


# =====================================================
# Environment Variables
# =====================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

PDF_FOLDER = "pdfs"
HASH_FILE = "uploaded_hashes.json"

os.makedirs(PDF_FOLDER, exist_ok=True)

if not os.path.exists(HASH_FILE):

    with open(HASH_FILE, "w") as f:

        json.dump({}, f)


# =====================================================
# Cached Embeddings
# =====================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =====================================================
# Cached Pinecone Client
# =====================================================


@st.cache_resource(show_spinner=False)
def get_pinecone_client():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    print("Indexes Before:")
    print(pc.list_indexes().names())

    if INDEX_NAME not in pc.list_indexes().names():

        print(f"Creating {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )


        while not pc.describe_index(
            INDEX_NAME
        ).status["ready"]:

            time.sleep(1)

    print("Indexes After:")
    print(pc.list_indexes().names())

    return pc

# =====================================================
# Cached Vector Store
# =====================================================

@st.cache_resource(show_spinner=False)
def get_vectorstore():

    # Ensure index exists first
    get_pinecone_client()

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=get_embeddings()
    )


# =====================================================
# Hash Utilities
# =====================================================

def load_hashes():

    with open(HASH_FILE, "r") as f:

        return json.load(f)


def save_hashes(data):

    with open(HASH_FILE, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )


# =====================================================
# SHA256
# =====================================================

def generate_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# =====================================================
# Duplicate Check
# =====================================================

def is_duplicate_pdf(file_hash):

    hashes = load_hashes()

    return file_hash in hashes


# =====================================================
# Save PDF
# =====================================================

def save_pdf(uploaded_file):

    file_bytes = uploaded_file.getvalue()

    file_hash = generate_file_hash(
        file_bytes
    )

    if is_duplicate_pdf(file_hash):

        return None, file_hash, True

    file_path = os.path.join(
        PDF_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:

        f.write(file_bytes)

    return file_path, file_hash, False


# =====================================================
# Chunk IDs
# =====================================================

def create_chunk_ids(
    filename,
    chunks
):

    ids = []

    for i in range(len(chunks)):

        ids.append(
            f"{filename}_{i}"
        )

    return ids


# =====================================================
# Ingest PDF
# =====================================================

def ingest_pdf(
    pdf_path,
    file_hash
):

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    filename = os.path.basename(
        pdf_path
    )

    for chunk in chunks:

        chunk.metadata["filename"] = filename

    vector_ids = create_chunk_ids(
        filename,
        chunks
    )

    vectorstore = get_vectorstore()

    vectorstore.add_documents(
        documents=chunks,
        ids=vector_ids
    )

    hashes = load_hashes()

    hashes[file_hash] = {
        "filename": filename,
        "vector_ids": vector_ids
    }

    save_hashes(hashes)

    return len(chunks)


# =====================================================
# Upload Workflow
# =====================================================

def process_uploaded_pdf(
    uploaded_file
):

    file_path, file_hash, duplicate = save_pdf(
        uploaded_file
    )

    if duplicate:

        return {
            "status": "duplicate",
            "message":
            "PDF already uploaded and indexed."
        }

    total_chunks = ingest_pdf(
        file_path,
        file_hash
    )

    return {
        "status": "success",
        "message":
        f"{uploaded_file.name} uploaded successfully.",
        "chunks": total_chunks
    }


# =====================================================
# Get Indexed PDFs
# =====================================================

def get_indexed_pdfs():

    hashes = load_hashes()

    files = []

    for value in hashes.values():

        files.append(
            value["filename"]
        )

    return sorted(files)


# =====================================================
# Delete PDF
# =====================================================

def delete_pdf(
    filename
):

    hashes = load_hashes()

    target_hash = None
    vector_ids = []

    for h, data in hashes.items():

        if data["filename"] == filename:

            target_hash = h

            vector_ids = data[
                "vector_ids"
            ]

            break

    if target_hash is None:

        return False

    # Delete Pinecone vectors

    pc = get_pinecone_client()

    index = pc.Index(
        INDEX_NAME
    )

    if vector_ids:

        index.delete(
            ids=vector_ids
        )

    # Delete PDF

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    if os.path.exists(pdf_path):

        os.remove(pdf_path)

    # Delete hash record

    del hashes[target_hash]

    save_hashes(hashes)

    return True

