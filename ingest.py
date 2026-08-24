import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

DOCUMENTS_PATH = "./documents"
CHROMA_PATH = "./chroma_db"

print("Loading documents....")

loader = DirectoryLoader(
    DOCUMENTS_PATH,
    glob="*.txt",
    loader_cls=TextLoader,
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")

print("Splitting documents....")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks. ")

print("Creating embeddings....")

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

print("Creating vector database...")

vector_store = Chroma(
    collection_name = "basic_rag",
    embedding_function = embeddings,
    persist_directory = CHROMA_PATH,
)


vector_store.add_documents(chunks)

print("Documents successfully stored in Chroma.")

