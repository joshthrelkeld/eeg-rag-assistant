import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

# Load PDF
loader = PyPDFLoader("data/eeg_paper.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# Embed and store in ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vectorstore created and saved to chroma_db/")