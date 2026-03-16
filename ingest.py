from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
 
DOC_PATH = "/home/vanshikabansal/Binary-AI/docs"
 
documents = []
 
for file in os.listdir(DOC_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOC_PATH, file))
        documents.extend(loader.load())
 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
 
chunks = text_splitter.split_documents(documents)
 
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
 
vector_db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="vector_db"
)
 
vector_db.persist()
 
print("Documents embedded successfully.")