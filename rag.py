import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

CHROMA_PATH = "./chroma_db"

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

Vector_store = Chroma(
    collection_name = "basic_rag",
    embedding_function=embeddings,
    persist_directory = CHROMA_PATH,
)

retriever = Vector_store.as_retriever(
    search_kwargs = {"k" : 3}
)

llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0
)

def ask_rag(question: str) -> str:

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say that you don't know.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content