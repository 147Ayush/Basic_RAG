# Basic RAG — Retrieval-Augmented Generation

A beginner-friendly implementation of **Basic Retrieval-Augmented Generation (RAG)** using **Python, LangChain, OpenAI Embeddings, ChromaDB, and an OpenAI LLM**.

This project demonstrates how an LLM can answer questions using information retrieved from a custom knowledge base instead of relying only on its pre-trained knowledge.

---

## 📌 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:

1. **Retrieval** — Find relevant information from a knowledge base.
2. **Augmentation** — Add the retrieved information to the LLM prompt as context.
3. **Generation** — The LLM generates an answer using that context.

### Basic RAG Flow

```text
                    DOCUMENT INGESTION
                    ==================

                       Documents
                           │
                           ▼
                   Document Loader
                           │
                           ▼
                    Text Splitter
                           │
                           ▼
                       Chunks
                           │
                           ▼
                   Embedding Model
                           │
                           ▼
                    Chroma Vector DB
                           │
                           │
                           │
                    USER QUERY
                           │
                           ▼
                    User Question
                           │
                           ▼
                   Similarity Search
                           │
                           ▼
                  Relevant Documents
                           │
                           ▼
                Context + User Query
                           │
                           ▼
                         LLM
                           │
                           ▼
                      Answer
```

---

# 🎯 Project Objective

The goal of this project is to understand the fundamental components of a RAG system by building one from scratch using modern LangChain APIs.

The project demonstrates:

* Document loading
* Text chunking
* Text embeddings
* Vector database storage
* Similarity-based retrieval
* Context construction
* LLM-based answer generation

This is the foundation for more advanced RAG architectures such as:

* Advanced RAG
* Hybrid RAG
* Agentic RAG
* Graph RAG
* Multimodal RAG

---

# 🏗️ Project Architecture

The project consists of two major pipelines.

## 1. Document Ingestion Pipeline

This pipeline prepares documents for retrieval.

```text
Documents
    │
    ▼
DirectoryLoader
    │
    ▼
TextLoader
    │
    ▼
RecursiveCharacterTextSplitter
    │
    ▼
Document Chunks
    │
    ▼
OpenAI Embeddings
    │
    ▼
ChromaDB
```

### Step-by-step

### Step 1 — Load Documents

Text files are loaded from the `documents/` directory.

```text
documents/
├── python.txt
├── langchain.txt
└── rag.txt
```

---

### Step 2 — Split Documents

Large documents are divided into smaller chunks.

```text
Large Document
      │
      ▼
┌──────────────┐
│   Chunk 1    │
├──────────────┤
│   Chunk 2    │
├──────────────┤
│   Chunk 3    │
└──────────────┘
```

This makes retrieval more efficient and allows the system to retrieve only the relevant portions of a document.

The project uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

---

### Step 3 — Generate Embeddings

Each chunk is converted into a numerical vector using the OpenAI embedding model.

```text
Text
 │
 ▼
Embedding Model
 │
 ▼
Vector
```

The project uses:

```text
text-embedding-3-small
```

Embeddings allow the system to compare the semantic similarity between a user question and document chunks.

---

### Step 4 — Store in Chroma

The document chunks and their embeddings are stored in **Chroma**, a vector database.

```text
Document Chunk
      +
Embedding
      +
Metadata
      ↓
   ChromaDB
```

---

# 2. Question Answering Pipeline

After ingestion, users can ask questions about the documents.

```text
User Question
      │
      ▼
Retriever
      │
      ▼
ChromaDB
      │
      ▼
Relevant Chunks
      │
      ▼
Build Context
      │
      ▼
Context + Question
      │
      ▼
OpenAI LLM
      │
      ▼
Final Answer
```

---

# 🧠 How Retrieval Works

Suppose the user asks:

```text
What is RAG?
```

The question is converted into an embedding.

```text
"What is RAG?"
      │
      ▼
Embedding Model
      │
      ▼
Question Vector
```

The vector is compared with vectors stored in Chroma.

The system retrieves the most relevant chunks.

```text
Question
   │
   ▼
Vector Similarity Search
   │
   ├── rag.txt       ← Highly relevant
   ├── langchain.txt ← Somewhat relevant
   └── python.txt    ← Less relevant
```

The top relevant documents are then provided to the LLM as context.

---

# 📂 Project Structure

```text
Basic_RAG/
│
├── documents/
│   ├── python.txt
│   ├── langchain.txt
│   └── rag.txt
│
├── chroma_db/
│
├── .env
├── .env.example
├── .gitignore
├── ingest.py
├── rag.py
├── main.py
├── requirements.txt
└── README.md
```

> `chroma_db/` is generated locally during ingestion and should not be committed to Git.

---

# 📄 File Description

| File / Directory   | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `documents/`       | Contains knowledge-base documents                       |
| `ingest.py`        | Loads, chunks, embeds, and stores documents             |
| `rag.py`           | Handles retrieval and LLM generation                    |
| `main.py`          | Provides the command-line interface                     |
| `requirements.txt` | Python dependencies                                     |
| `.env`             | Stores API credentials locally                          |
| `.env.example`     | Example environment configuration                       |
| `.gitignore`       | Prevents sensitive/generated files from being committed |
| `chroma_db/`       | Local Chroma vector database                            |
| `README.md`        | Project documentation                                   |

---

# 🔧 Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Programming language                 |
| LangChain        | RAG application framework            |
| LangChain OpenAI | OpenAI LLM and embedding integration |
| OpenAI           | LLM and embedding models             |
| Chroma           | Vector database                      |
| Python Dotenv    | Environment variable management      |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/147Ayush/Basic_RAG.git
```

Move into the project:

```bash
cd Basic_RAG
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

# 📥 Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install the latest packages manually:

```bash
pip install -U langchain langchain-openai langchain-chroma langchain-text-splitters langchain-community python-dotenv
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

You can use `.env.example` as a template.

### Important

Never commit your real API key to GitHub.

The `.env` file should be included in `.gitignore`.

---

# ▶️ Running the Project

The project has two main stages:

```text
1. Ingestion
2. Question Answering
```

## Step 1 — Ingest Documents

Run:

```bash
python ingest.py
```

This will:

```text
Load documents
      ↓
Split documents
      ↓
Create embeddings
      ↓
Create/update Chroma database
```

You should see output similar to:

```text
Loading documents...
Loaded documents.
Splitting documents...
Created chunks.
Creating embeddings...
Creating vector database...
Documents successfully stored in Chroma.
```

---

## Step 2 — Start the RAG Application

Run:

```bash
python main.py
```

You can now ask questions.

Example:

```text
You: What is RAG?
```

The system retrieves relevant information and generates an answer.

Example:

```text
Assistant: RAG stands for Retrieval-Augmented Generation.
It combines information retrieval with a large language
model. The system retrieves relevant information from a
knowledge base and provides it to the LLM as context.
```

---

# 🧪 Example Questions

You can ask:

```text
What is RAG?
```

```text
What is LangChain?
```

```text
What programming language is used in this project?
```

```text
What are the benefits of RAG?
```

```text
Why does RAG use embeddings?
```

---

# 🔍 Retriever Configuration

The project creates a retriever from Chroma:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
```

The value:

```text
k = 3
```

means the system retrieves the top 3 relevant document chunks.

Conceptually:

```text
User Question
      ↓
Chroma Search
      ↓
Top 3 Relevant Chunks
      ↓
LLM
```

---

# 🤖 LLM Configuration

The project uses an OpenAI chat model:

```python
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)
```

`temperature=0` is used to make responses more deterministic.

---

# 🧩 Core RAG Components

This project teaches the following RAG components:

### 1. Document Loader

Loads knowledge from files.

```python
DirectoryLoader
TextLoader
```

### 2. Text Splitter

Breaks documents into manageable chunks.

```python
RecursiveCharacterTextSplitter
```

### 3. Embedding Model

Converts text into vectors.

```python
OpenAIEmbeddings
```

### 4. Vector Database

Stores and searches embeddings.

```python
Chroma
```

### 5. Retriever

Finds relevant chunks.

```python
vector_store.as_retriever()
```

### 6. LLM

Generates the final response.

```python
ChatOpenAI
```

---

# 🔄 Complete Data Flow

```text
                  ┌─────────────────┐
                  │    Documents    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Document Loader │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Text Splitter  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Embeddings    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Chroma      │
                  │  Vector Store   │
                  └────────┬────────┘
                           │
                           │
                           │
User Question ─────────────┤
                           ▼
                  ┌─────────────────┐
                  │    Retriever    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Relevant Context│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      LLM        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Final Answer   │
                  └─────────────────┘
```

---

# 🆚 Traditional LLM vs RAG

## Traditional LLM

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

The model primarily relies on knowledge encoded during training.

---

## RAG

```text
User Question
      ↓
   Retriever
      ↓
Knowledge Base
      ↓
Relevant Context
      ↓
     LLM
      ↓
   Answer
```

RAG allows the application to provide external knowledge to the LLM at query time.

---

# ✅ Advantages of Basic RAG

* Can use private documents
* Reduces dependency on model training knowledge
* Knowledge can be updated without retraining the LLM
* Provides relevant context to the LLM
* Works well for document-based question answering
* Relatively simple to implement

---

# ⚠️ Limitations of Basic RAG

Basic RAG also has limitations.

### Retrieval Problems

If the wrong chunks are retrieved, the LLM may receive poor context.

### Chunking Problems

Bad chunk sizes can cause:

* Missing information
* Too much irrelevant information
* Broken context

### Semantic Search Limitations

Vector search may not perform well with:

* Exact error codes
* Product IDs
* Names
* Numbers
* Keywords

This is one reason why **Hybrid RAG** is useful.

### No Advanced Reranking

Basic RAG retrieves the top results but does not use a dedicated reranking model.

### No Agentic Decision Making

The system doesn't decide dynamically which tools or retrieval strategies to use.

These limitations will be addressed in the next RAG architectures.

---

# 🚀 Future Improvements

This project is the foundation for more advanced RAG systems.

Planned progression:

```text
Basic RAG
    │
    ▼
Advanced RAG
    │
    ├── Better Chunking
    ├── Metadata Filtering
    ├── Query Transformation
    └── Reranking
    │
    ▼
Hybrid RAG
    │
    ├── Vector Search
    └── Keyword Search / BM25
    │
    ▼
Agentic RAG
    │
    ▼
Graph RAG
    │
    ▼
Multimodal RAG
```

---

# 📚 Learning Objectives

After completing this project, you should understand:

* What RAG is
* Why RAG is needed
* How document ingestion works
* What document chunks are
* What embeddings are
* How vector databases work
* What similarity search does
* What a retriever is
* How retrieved context is passed to an LLM
* How a basic RAG pipeline is implemented using LangChain

---

# 🎓 Key Concepts

```text
RAG
│
├── Documents
│
├── Document Loading
│
├── Chunking
│
├── Embeddings
│
├── Vector Database
│
├── Similarity Search
│
├── Retrieval
│
├── Context
│
└── Generation
```

The most important concept to remember:

```text
RAG = Retrieve relevant information
      +
      Augment the LLM with that information
      +
      Generate an answer
```

---

# 🔐 Security

Never commit API keys or secrets.

Use:

```text
.env
```

for local credentials and:

```text
.env.example
```

for the public configuration template.

The `.gitignore` file should exclude:

```text
.env
.venv/
__pycache__/
chroma_db/
.idea/
```

---

# 📌 Project Status

**Status:** ✅ Basic RAG Completed

This repository represents the first implementation in a planned series of RAG architectures.

### RAG Learning Roadmap

* [x] Basic / Naive RAG
* [ ] Advanced RAG
* [ ] Hybrid RAG
* [ ] Agentic RAG
* [ ] Graph RAG
* [ ] Multimodal RAG

---

# 👨‍💻 Author

**Ayush Soni**

Learning and building projects around:

* Python
* AIOps
* MLOps
* Generative AI
* LLMs
* AI Agents
* LangChain
* LangGraph
* RAG

---

# ⭐ If You Find This Useful

If this project helps you understand RAG, consider giving the repository a ⭐ on GitHub.

The project will continue to evolve as more advanced RAG architectures are implemented.
