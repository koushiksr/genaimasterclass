from langchain_chroma import Chroma
import os
import tiktoken
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
import plotly.graph_objects as go

MODEL = "gpt-4.1-mini"
db_name = "vector_db"
chunk_size = 1000
chunk_overlap = 200

# ---------------- Load Documents ---------------- #

KNOWLEDGE_BASE = Path("/Users/koushiksr/Documents/workspace/allrepos_microdegree/genaimasterclass/knowledge-base")
loader = DirectoryLoader(path=str(KNOWLEDGE_BASE),glob="**/*.md",loader_cls=TextLoader)

documents = loader.load()
# print(f"Loaded {len(documents)} markdown files")

# ---------------- Split Documents to chunks ---------------- #

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(documents)
# print(f"Created {len(chunks)} chunks")
# print(chunks)

# ---------------- Embeddings using hugging face ---------------- #

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(db_name):
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name,
    )

print(f"Vector store created successfully {vectorstore._collection.count()}")
data = vectorstore.get(include=["embeddings"])
# ---------------- visualize vectore in 2d 3d ----------------------


embeddings_array = np.array(data["embeddings"])

# ---------------- Reduce dimensionality using t-SNE for 2D and 3D visualization ----------------

tsne_2d = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000,
)

embeddings_2d = tsne_2d.fit_transform(embeddings_array)

tsne_3d = TSNE(
    n_components=3,
    perplexity=30,
    random_state=42,
    max_iter=1000,
)

embeddings_3d = tsne_3d.fit_transform(embeddings_array)
fig = go.Figure(
    go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode="markers",
        text=[doc.metadata.get("source", "") for doc in chunks],
    )
)

fig.show(renderer="browser")
fig = go.Figure(
    go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode="markers",
        text=[doc.metadata.get("source", "") for doc in chunks],
    )
)

fig.show(renderer="browser")









# entire_knowledge_base = ""

# for file_path in files:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         entire_knowledge_base += f.read()
#         entire_knowledge_base += "\n\n"

# print(f"Total characters in knowledge base: {len(entire_knowledge_base):,}")



# encoding = tiktoken.encoding_for_model(MODEL)
# tokens = encoding.encode(entire_knowledge_base)
# token_count = len(tokens)
# print(f"Total tokens is {MODEL}: {token_count}")


# #LangChain Framework
# #Chunking
# #Embedding
# #Vector Database -- Chroma 
