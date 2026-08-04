import os
import glob
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from dotenv import load_dotenv

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        doc_type = os.path.basename(folder)
        md_files = glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)
        for file_path in md_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "doc_type": doc_type}
            )
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    if chunks:
        create_embeddings(chunks)
        print("Ingestion complete")
    else:
        print("No chunks to embed")
