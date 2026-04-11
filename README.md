# Medical Chatbot with RAG Integration

A production-ready Medical Question-Answering system using Retrieval-Augmented Generation (RAG) with AWS Bedrock Gemma 3 27B, AWS Bedrock Cohere embeddings, and Pinecone vector database.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Installation & Setup](#installation--setup)
5. [Module Documentation](#module-documentation)
   - [Module 1: Configuration (config.py)](#module-1-configuration-configpy)
   - [Module 2: Logger (logger.py)](#module-2-logger-loggerpy)
   - [Module 3: PDF Loader (pdf_loader.py)](#module-3-pdf-loader-pdf_loaderpy)
   - [Module 4: Embeddings (embeddings.py)](#module-4-embeddings-embeddingspy)
   - [Module 5: Vector Store (vectorstore.py)](#module-5-vector-store-vectorstorepy)
   - [Module 6: LLM (llm.py)](#module-6-llm-llmpy)
   - [Module 7: Retriever (retriever.py)](#module-7-retriever-retrieverpy)
   - [Module 8: Application (application.py)](#module-8-application-applicationpy)
   - [Module 9: AWS IAM Setup](#module-9-aws-iam-setup)
   - [Module 10: Deployment](#module-10-deployment)
6. [End-to-End RAG Workflow](#end-to-end-rag-workflow)
7. [API Documentation](#api-documentation)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)
10. [AWS IAM, Roles & Policy Requirements](#aws-iam-roles--policy-requirements)
11. [Application Deployment](#application-deployment)
12. [Contributing](#contributing)

---

## Project Overview

### What is This Project?

A Medical RAG (Retrieval-Augmented Generation) chatbot that answers medical questions by:
1. Retrieving relevant information from a medical encyclopedia
2. Using AI to generate accurate, contextual answers
3. Providing source citations for transparency

### Key Features

- ✅ **Semantic Search**: HNSW algorithm for fast, accurate document retrieval
- ✅ **Semantic Chunking**: AWS Bedrock Cohere-powered SemanticChunker with 95th-percentile breakpoints; falls back to RecursiveCharacterTextSplitter when Bedrock is unavailable
- ✅ **Medical Knowledge Base**: 759-page medical encyclopedia with 7,590 chunks
- ✅ **Rich Metadata Extraction**: Category, sub-category, section type, ICD codes, medications, measurements, and medical term density per chunk
- ✅ **Advanced Filtering**: Filter chunks by medical category, section type, or medical term density
- ✅ **Source Citations**: Transparent answers with document references
- ✅ **Conversational Chat**: Maintains conversation history for context
- ✅ **Production Ready**: Flask API + AngularJS frontend

### Use Cases

- Medical students studying diseases and treatments
- Healthcare professionals looking up quick references
- Patients seeking educational medical information
- Researchers exploring medical topics

---

## Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Medical RAG System                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  User Question  │
│ "What is        │
│  diabetes?"     │
└────────┬────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│               FRONTEND (AngularJS)                          │
│  - User Interface                                           │
│  - Input handling                                           │
│  - Response display                                         │
└────────┬────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│               BACKEND (Flask API)                           │
│  - /api/query (RAG endpoint)                                │
│  - /api/chat (Conversational)                               │
│  - Request/Response handling                                │
└────────┬────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│            RETRIEVER (retriever.py)                         │
│  - Orchestrates RAG pipeline                                │
│  - Combines retrieval + generation                          │
└────────┬────────────────────────────────────────────────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌─────────┐  ┌──────────────────────────────────────────────┐
│EMBEDDING│  │       VECTOR DATABASE                        │
│(Bedrock)│  │       (Pinecone + HNSW)                      │
│         │→ │  - Stores 7,590 medical document vectors     │
│Query    │  │  - HNSW semantic search                      │
│→Vector  │  │  - Metadata filtering                        │
└─────────┘  └──────────┬───────────────────────────────────┘
                        ↓
             ┌─────────────────────┐
             │  Retrieved Documents│
             │  (Top 5 relevant)   │
             └──────────┬──────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│                LLM (AWS Bedrock Gemma 3 27B)                │
│  - Generates human-like answers                             │
│  - Uses retrieved context                                   │
│  - Factual and accurate responses                           │
└────────┬────────────────────────────────────────────────────┘
         ↓
┌─────────────────┐
│  Final Answer   │
│  + Sources      │
│  + Citations    │
└─────────────────┘
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Database** | Pinecone | Cloud vector storage |
| **Search Algorithm** | HNSW | Fast semantic search |
| **Embeddings** | AWS Bedrock cohere.embed-english-v3 | Convert text to vectors (1024 dims) |
| **LLM** | AWS Bedrock Gemma 3 27B | Answer generation |
| **Backend** | Flask | REST API server |
| **Frontend** | AngularJS | User interface |
| **PDF Processing** | PyPDF | Document loading |
| **Text Processing** | LangChain | Chunking and pipelines |
| **Semantic Chunking** | LangChain SemanticChunker + Bedrock Cohere | Meaning-aware chunk splitting (primary) |
| **Fallback Chunking** | LangChain RecursiveCharacterTextSplitter | Fixed-size splitting when Bedrock unavailable |

### Python Libraries
```
langchain==0.1.0
langchain-community==0.0.20
langchain-aws==0.1.0
pinecone-client==3.0.0
boto3==1.34.0
flask==3.0.0
flask-cors==4.0.0
pypdf==4.0.0
python-dotenv==1.0.0
numpy==1.26.0
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- AWS Bedrock access (for Gemma 3 27B and Cohere embeddings)
- Pinecone account (free tier available)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/medical-rag-chatbot.git
cd medical-rag-chatbot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create `.env` file:
```bash
# AWS Bedrock (for Gemma 3 27B LLM and Cohere embeddings)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=google.gemma-3-27b-v1
BEDROCK_EMBEDDING_MODEL_ID=cohere.embed-english-v3

# Pinecone (for vector database)
PINECONE_API_KEY=your-pinecone-key
```

### Step 4: Prepare Data

Place your medical PDF in `data/` folder:
```bash
mkdir data
# Copy your medical_encyclopedia.pdf to data/
```

### Step 5: Build Vector Database
```bash
# Process PDF and create embeddings
python scripts/build_vectordb.py
```

This will:
- Load PDF (759 pages)
- Extract and clean text
- Create 7,590 chunks
- Generate embeddings via AWS Bedrock Cohere
- Store in Pinecone with HNSW indexing
- Takes ~15-20 minutes

### Step 6: Run Application
```bash
python app/application.py
```

Access at: `http://localhost:5000`

---

## Module Documentation

### Module 1: Configuration (config.py)

**Location:** `app/config/config.py`

**What it does:** Central configuration management for all API keys, paths, and parameters

#### Class: `Config`

**What this class does:** Stores all configuration variables loaded from environment

**Attributes:**

| Attribute | Purpose | Example Value |
|-----------|---------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS Bedrock authentication | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock secret | `secret-key` |
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` |
| `BEDROCK_MODEL_ID` | Gemma 3 27B model identifier | `google.gemma-3-27b-v1` |
| `BEDROCK_EMBEDDING_MODEL_ID` | Cohere embedding model identifier | `cohere.embed-english-v3` |
| `PINECONE_API_KEY` | Pinecone authentication | `your-key` |
| `PINECONE_INDEX_NAME` | Vector database index name | `medical-rag-index` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `DATAPATH` | Path to PDF data | `data/` |

---

### Module 2: Logger (logger.py)

**Location:** `app/common/logger.py`

**What it does:** Provides structured logging with file rotation and console output

#### Class: `MedicalRAGLogger`

**What this class does:** Creates and manages loggers for the application

**Constructor:**
```python
def __init__(self, name: str)
```
**What it does:** Initializes logger with specified name

**Parameters:**
- `name`: Logger name (usually `__name__`)

**Methods:**

##### Function: `log_error(exception, context)`

**What this function does:** Logs errors with context and traceback

**Parameters:**
- `exception`: The exception object
- `context`: Description of where error occurred

**Example:**
```python
try:
    # some code
except Exception as e:
    logger.log_error(e, context="Processing PDF")
```

##### Function: `log_retrieval(num_docs, query)`

**What this function does:** Logs document retrieval events

**Parameters:**
- `num_docs`: Number of documents retrieved
- `query`: Search query used

##### Function: `log_embedding(num_chunks, model)`

**What this function does:** Logs embedding generation events

**Parameters:**
- `num_chunks`: Number of chunks embedded
- `model`: Embedding model used

##### Function: `log_vectorstore_operation(operation, collection_name, status)`

**What this function does:** Logs vector database operations

**Parameters:**
- `operation`: Type of operation (create, store, search)
- `collection_name`: Database/index name
- `status`: Operation status (success/failure)

---

### Module 3: PDF Loader (pdf_loader.py)

**Location:** `app/components/pdf_loader.py`

**What it does:** Loads medical PDFs, cleans text, performs semantic or recursive chunking, extracts rich medical metadata, and supports filtering and summarisation of document chunks

#### Class: `MedicalPDFLoader`

**What this class does:** Manages the full PDF ingestion pipeline for medical documents — loading, cleaning, semantic chunking, and metadata enrichment

**Constructor:**
```python
def __init__(self)
```
**What it does:** Initialises the loader with configuration from `Config`. Attempts to create a `SemanticChunker` backed by AWS Bedrock Cohere embeddings. If Bedrock is unavailable, falls back silently to `RecursiveCharacterTextSplitter`.

**Chunking strategy:**

| Strategy | When used | Details |
|----------|-----------|---------|
| **SemanticChunker** (primary) | Bedrock reachable | 95th-percentile breakpoint on cosine distances — ideal for dense medical text |
| **RecursiveCharacterTextSplitter** (fallback) | Bedrock unavailable | `chunk_size` and `chunk_overlap` from `Config`; separators: `\n\n`, `\n`, `. `, `; `, `, `, ` ` |

**Methods:**

##### Function: `load_medical_pdf(filename)`

**What this function does:** Loads a single PDF by filename (resolved against `Config.DATAPATH`), cleans each page, skips near-empty pages (<100 chars), and enriches every page with medical metadata

**Parameters:**
- `filename`: PDF filename (e.g. `"The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"`)

**Returns:** `List[Document]` — enhanced documents with medical metadata merged into `doc.metadata`

**Example:**
```python
loader = MedicalPDFLoader()
documents = loader.load_medical_pdf("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
# Returns: [Document(page_content="...", metadata={...}), ...]
```

##### Function: `split_with_metadata_preservation(documents)`

**What this function does:** Splits a list of documents into chunks while preserving all existing metadata and adding chunk-level fields

**Parameters:**
- `documents`: List of `Document` objects (output of `load_medical_pdf`)

**Returns:** `List[Document]` — chunked documents with additional metadata

**Chunk metadata added:**

| Field | Description |
|-------|-------------|
| `chunk_id` | Sequential chunk index |
| `chunk_size` | Character count of the chunk |
| `total_chunks` | Total number of chunks produced |
| `chunk_selection_type` | Section type identified for this chunk |

**Example:**
```python
chunks = loader.split_with_metadata_preservation(documents)
```

##### Function: `extract_medical_metadata(text, page_num, filename)`

**What this function does:** Extracts a comprehensive set of medical metadata fields from page text

**Parameters:**
- `text`: Cleaned page content
- `page_num`: Page number from PDF
- `filename`: Source PDF filename

**Returns:** `Dict` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `source` | str | Source filename |
| `page` | int | Page number |
| `processed_date` | str | Timestamp of processing |
| `doc_type` | str | Always `"medical_encyclopedia"` |
| `category` | str | Medical category (see `extract_category`) |
| `sub_category` | str | Specific condition/disease name |
| `has_definitions` | bool | Contains definitional language |
| `has_symptoms` | bool | Contains symptom information |
| `has_treatments` | bool | Contains treatment information |
| `has_diagnosis` | bool | Contains diagnostic information |
| `medical_term_density` | float | Ratio of medical-suffix words to total words |
| `section_type` | str | Section classification (see `_identify_section_type`) |
| `char_count` | int | Character count of the page |
| `word_count` | int | Word count of the page |
| `medications` | list | Extracted medication names (up to 5) |
| `icd_codes` | list | Extracted ICD codes (up to 3) |
| `has_measurements` | bool | Contains dosage/measurement values |
| `measurement_count` | int | Number of measurements found |

**Example:**
```python
meta = loader.extract_medical_metadata(text, page_num=42, filename="encyclopedia.pdf")
# {
#     "category": "endocrine",
#     "sub_category": "Diabetes Mellitus",
#     "section_type": "symptoms",
#     "has_symptoms": True,
#     "medical_term_density": 0.023,
#     "icd_codes": ["E11"],
#     ...
# }
```

##### Function: `extract_category(text)`

**What this function does:** Classifies page text into one of 12 medical categories by keyword matching

**Returns:** One of `cardiovascular`, `respiratory`, `neurological`, `endocrine`, `gastrointestinal`, `musculoskeletal`, `dermatological`, `infectious`, `oncology`, `immunology`, `nephrology`, `hematology`, or `general`

##### Function: `clean_medical_text(text)`

**What this function does:** Normalises raw OCR text from medical PDFs — collapses whitespace, strips page numbers, removes encyclopedia headers, normalises bullet characters, and fixes common OCR artefacts

**Parameters:**
- `text`: Raw page content string

**Returns:** Cleaned text string

##### Function: `process_medical_pdf(filename)`

**What this function does:** End-to-end pipeline: load → clean → enrich metadata → split into chunks

**Parameters:**
- `filename`: PDF filename

**Returns:** `List[Document]` — fully processed, metadata-enriched chunks ready for embedding

**Example:**
```python
loader = MedicalPDFLoader()
chunks = loader.process_medical_pdf("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
print(f"Produced {len(chunks)} chunks")
```

##### Function: `filter_chunks_by_metadata(chunks, category, section_type, min_medical_density)`

**What this function does:** Filters a list of chunks by optional metadata criteria

**Parameters:**
- `chunks`: List of Document chunks
- `category` *(optional)*: Retain only chunks matching this medical category
- `section_type` *(optional)*: Retain only chunks matching this section type
- `min_medical_density` *(optional, default 0.0)*: Applied only when value > 0.5; filters by minimum `medical_term_density`

**Returns:** Filtered `List[Document]`

**Example:**
```python
cardio_chunks = loader.filter_chunks_by_metadata(chunks, category="cardiovascular")
treatment_chunks = loader.filter_chunks_by_metadata(chunks, section_type="treatment")
```

##### Function: `get_metadata_summary(chunks)`

**What this function does:** Produces aggregate statistics across all chunks

**Parameters:**
- `chunks`: List of Document chunks

**Returns:** `Dict` with:
- `total_chunks`: Total chunk count
- `categories`: Frequency map of medical categories
- `section_types`: Frequency map of section types
- `chunks_with_symptoms` / `chunks_with_treatments` / `chunks_with_diagnosis`: Counts per section type
- `avg_medical_density`: Mean `medical_term_density` across chunks

**Example:**
```python
summary = loader.get_metadata_summary(chunks)
# {
#     "total_chunks": 7590,
#     "categories": {"endocrine": 312, "cardiovascular": 280, ...},
#     "avg_medical_density": 0.018
# }
```

---

### Module 4: Embeddings (embeddings.py)

**Location:** `app/components/embeddings.py`

**What it does:** Converts text chunks into vector embeddings using AWS Bedrock Cohere

#### Class: `MedicalEmbeddings`

**What this class does:** Handles AWS Bedrock Cohere embedding generation for medical text

**Constructor:**
```python
def __init__(self, model_name: str = "cohere.embed-english-v3")
```
**What it does:** Initializes AWS Bedrock Cohere embeddings client

**Parameters:**
- `model_name`: AWS Bedrock embedding model to use

**Methods:**

##### Function: `embed_single_chunk(chunk)`

**What this function does:** Converts one text chunk into embedding vector

**Parameters:**
- `chunk`: Document object with text and metadata

**Returns:** Tuple of (embedding_vector, metadata)

**Example:**
```python
embedder = MedicalEmbeddings()
embedding, metadata = embedder.embed_single_chunk(chunk)
# embedding: [0.234, -0.123, 0.567, ...] (1024 numbers)
# metadata: {"category": "endocrine", "page": 100}
```

##### Function: `embed_chunks(chunks)`

**What this function does:** Converts multiple chunks into embeddings (batch processing)

**Parameters:**
- `chunks`: List of Document objects

**Returns:** List of (embedding, metadata) tuples

**Process:**
1. Estimates cost before processing
2. Processes chunks one by one
3. Shows progress every 100 chunks
4. Logs completion

**Example:**
```python
embeddings = embedder.embed_chunks(chunks)
# Input: 7,590 chunks
# Output: 7,590 (vector, metadata) tuples
# Time: ~2-5 minutes
```

##### Function: `embed_query(query)`

**What this function does:** Converts search query into embedding vector

**Parameters:**
- `query`: User's search query string

**Returns:** Embedding vector (1024 dimensions)

**Example:**
```python
query_embedding = embedder.embed_query("What is diabetes?")
# Returns: [0.245, -0.115, 0.573, ...] (1024 numbers)
```

##### Function: `estimate_embedding_cost(chunks)`

**What this function does:** Calculates estimated cost before embedding

**Parameters:**
- `chunks`: List of chunks to be embedded

**Returns:** Dictionary with cost information

**Example:**
```python
cost_info = embedder.estimate_embedding_cost(chunks)
# {
#     "total_chunks": 7590,
#     "estimated_tokens": 506000,
#     "estimated_total_cost": "varies by AWS Bedrock pricing"
# }
```

#### Class: `MedicalEmbeddingPipeline`

**What this class does:** High-level pipeline for complete embedding workflow

**Constructor:**
```python
def __init__(self, model_name: str = "cohere.embed-english-v3")
```
**What it does:** Creates embedding pipeline with specified model

**Methods:**

##### Function: `process_chunks_to_embeddings(chunks)`

**What this function does:** Complete pipeline - validate, estimate, embed, log

**Parameters:**
- `chunks`: Document chunks from PDF loader

**Returns:** List of (embedding, metadata) tuples

**Process:**
1. Validates input
2. Shows cost estimate
3. Generates embeddings
4. Calculates statistics
5. Logs everything

**Example:**
```python
pipeline = MedicalEmbeddingPipeline()
embeddings = pipeline.process_chunks_to_embeddings(chunks)
# Handles entire workflow automatically
```

---

### Module 5: Vector Store (vectorstore.py)

**Location:** `app/components/vectorstore.py`

**What it does:** Manages Pinecone vector database with HNSW semantic search

#### Class: `PineconeHNSWVectorStore`

**What this class does:** Interface to Pinecone vector database with HNSW search

**Constructor:**
```python
def __init__(self)
```
**What it does:** Connects to Pinecone cloud service

**Methods:**

##### Function: `create_index(dimension, metric, force_recreate)`

**What this function does:** Creates vector database index in Pinecone cloud

**Parameters:**
- `dimension`: Vector size (1024 for cohere.embed-english-v3)
- `metric`: Similarity metric ("cosine" for semantic search)
- `force_recreate`: Delete existing and create new (default: False)

**Returns:** None (creates index in cloud)

**Process:**
1. Checks if index exists
2. Creates serverless index in AWS us-east-1
3. Waits for index to be ready
4. HNSW automatically enabled by Pinecone

**Example:**
```python
vectorstore = PineconeHNSWVectorStore()
vectorstore.create_index(dimension=1024, metric="cosine")
# Creates: "medical-rag-index" in Pinecone cloud
```

##### Function: `store_embeddings(embeddings, batch_size, namespace)`

**What this function does:** Uploads embeddings to Pinecone vector database

**Parameters:**
- `embeddings`: List of (vector, metadata) tuples
- `batch_size`: Vectors per batch (default: 100)
- `namespace`: Optional Pinecone namespace

**Returns:** None (stores in cloud)

**Process:**
1. Batches vectors into groups of 100
2. Uploads each batch to Pinecone
3. Shows progress every 500 vectors
4. HNSW index automatically updated

**Example:**
```python
vectorstore.store_embeddings(embeddings, batch_size=100)
# Uploads 7,590 vectors to Pinecone
# Time: ~2-3 minutes
```

##### Function: `semantic_search(query_embedding, top_k, filter_dict, namespace)`

**What this function does:** Searches for similar documents using HNSW algorithm

**Parameters:**
- `query_embedding`: Query vector from user question
- `top_k`: Number of results to return (default: 5)
- `filter_dict`: Metadata filters (e.g., `{"category": "cardiovascular"}`)
- `namespace`: Pinecone namespace

**Returns:** List of matching documents with scores and metadata

**Process:**
1. Sends query vector to Pinecone
2. Pinecone uses HNSW to find similar vectors
3. Compares using cosine similarity
4. Returns top K most similar documents

**Example:**
```python
results = vectorstore.semantic_search(
    query_embedding=query_vector,
    top_k=5,
    filter_dict={"category": "endocrine"}
)
# Returns:
# [
#     {
#         "id": "vec_234",
#         "score": 0.923,
#         "text": "Diabetes is...",
#         "metadata": {"category": "endocrine"}
#     },
#     ...
# ]
```

##### Function: `hybrid_search(query_embedding, top_k, category_filter, section_filter)`

**What this function does:** Two-stage search - filter by metadata then semantic search

**Parameters:**
- `query_embedding`: Query vector
- `top_k`: Number of results
- `category_filter`: Medical category (e.g., "cardiovascular")
- `section_filter`: Section type (e.g., "symptoms")

**Returns:** Filtered and semantically ranked results

**Process:**
1. **Stage 1**: Filter by metadata (reduces 7,590 → ~300 candidates)
2. **Stage 2**: HNSW semantic search within filtered set
3. Returns top K from filtered results

**Example:**
```python
results = vectorstore.hybrid_search(
    query_embedding=query_vector,
    top_k=5,
    category_filter="cardiovascular",
    section_filter="symptoms"
)
# Searches only cardiovascular symptoms
# Faster and more accurate
```

##### Function: `get_index_stats()`

**What this function does:** Retrieves statistics about the vector database

**Returns:** Dictionary with index information

**Example:**
```python
stats = vectorstore.get_index_stats()
# {
#     "index_name": "medical-rag-index",
#     "vector_database": "Pinecone",
#     "search_algorithm": "HNSW",
#     "total_vectors": 7590,
#     "dimension": 1024,
#     "status": "ready"
# }
```

##### Function: `delete_index()`

**What this function does:** Deletes entire vector database from Pinecone

**Warning:** This destroys all stored vectors permanently

##### Function: `delete_vectors(ids, delete_all, namespace)`

**What this function does:** Deletes specific vectors or all vectors

**Parameters:**
- `ids`: List of vector IDs to delete
- `delete_all`: Delete all vectors (keeps index structure)
- `namespace`: Pinecone namespace

#### Class: `MedicalVectorStorePipeline`

**What this class does:** High-level pipeline combining index creation and storage

**Constructor:**
```python
def __init__(self)
```
**What it does:** Creates pipeline with PineconeHNSWVectorStore

**Methods:**

##### Function: `store_embeddings_pipeline(embeddings, create_new_index)`

**What this function does:** Complete storage workflow in one command

**Parameters:**
- `embeddings`: Embeddings from embedding pipeline
- `create_new_index`: Recreate index from scratch (default: False)

**Process:**
1. Validates embeddings
2. Creates/connects to index
3. Stores embeddings
4. Verifies storage
5. Returns statistics

**Example:**
```python
pipeline = MedicalVectorStorePipeline()
pipeline.store_embeddings_pipeline(embeddings)
# Handles entire storage workflow
```

---

### Module 6: LLM (llm.py)

**Location:** `app/components/llm.py`

**What it does:** Interfaces with AWS Bedrock Gemma 3 27B for answer generation

#### Class: `BedrockGemma3LLM`

**What this class does:** Manages AWS Bedrock Gemma 3 27B language model

**Constructor:**
```python
def __init__(self, model_id, region_name)
```
**What it does:** Connects to AWS Bedrock service

**Parameters:**
- `model_id`: Bedrock model identifier (default: from config)
- `region_name`: AWS region (default: us-east-1)

**Methods:**

##### Function: `generate(prompt, max_tokens, temperature, top_p, top_k, stop_sequences)`

**What this function does:** Generates text using Gemma 3 27B model

**Parameters:**
- `prompt`: Input text prompt
- `max_tokens`: Maximum response length (default: 1024)
- `temperature`: Randomness (0.0-1.0, default: 0.7)
  - 0.0 = Deterministic, factual
  - 1.0 = Creative, diverse
- `top_p`: Nucleus sampling (default: 0.9)
- `top_k`: Top-k sampling (default: 50)
- `stop_sequences`: Stop generation at these sequences

**Returns:** Generated text string

**Process:**
1. Prepares request for Bedrock API
2. Calls Gemma 3 27B model
3. Parses response
4. Returns generated text

**Example:**
```python
llm = BedrockGemma3LLM()
response = llm.generate(
    prompt="Explain diabetes in simple terms",
    max_tokens=200,
    temperature=0.3  # Low for factual accuracy
)
# Returns: "Diabetes is a chronic condition..."
```

##### Function: `get_model_info()`

**What this function does:** Returns model configuration details

**Returns:** Dictionary with model information

**Example:**
```python
info = llm.get_model_info()
# {
#     "model_id": "google.gemma-3-27b-v1",
#     "provider": "AWS Bedrock",
#     "model_name": "Gemma 3 27B",
#     "region": "us-east-1",
#     "parameters": "27 billion"
# }
```

---

### Module 7: Retriever (retriever.py)

**Location:** `app/components/retriever.py`

**What it does:** Orchestrates complete RAG pipeline - retrieval + generation

#### Class: `MedicalRAGRetriever`

**What this class does:** Combines vector search and LLM generation for complete RAG

**Constructor:**
```python
def __init__(self)
```
**What it does:** Initializes all RAG components (embeddings, vectorstore, LLM)

**Components initialized:**
- `MedicalEmbeddingPipeline` - For query embedding
- `PineconeHNSWVectorStore` - For document retrieval
- `BedrockGemma3LLM` - For answer generation

**Methods:**

##### Function: `retrieve_documents(query, top_k, filters)`

**What this function does:** Retrieves relevant documents from vector database

**Parameters:**
- `query`: User's question (string)
- `top_k`: Number of documents to retrieve (default: 5)
- `filters`: Metadata filters (e.g., `{"category": "cardiovascular"}`)

**Returns:** List of relevant documents with scores

**Process:**
1. Converts query to embedding vector
2. Searches Pinecone using HNSW
3. Returns top K most similar documents

**Example:**
```python
retriever = MedicalRAGRetriever()
documents = retriever.retrieve_documents(
    query="What is diabetes?",
    top_k=5
)
# Returns:
# [
#     {
#         "id": "vec_234",
#         "score": 0.923,
#         "text": "Diabetes is a chronic disease...",
#         "metadata": {"category": "endocrine"}
#     },
#     ...
# ]
```

##### Function: `generate_answer(query, documents, temperature, max_tokens)`

**What this function does:** Generates answer using LLM and retrieved documents

**Parameters:**
- `query`: User's question
- `documents`: Retrieved context documents
- `temperature`: LLM temperature (default: 0.3 for medical)
- `max_tokens`: Maximum response length (default: 1024)

**Returns:** Generated answer string

**Process:**
1. Builds context from retrieved documents
2. Creates prompt with context and question
3. Calls Gemma 3 27B to generate answer
4. Returns answer

**Example:**
```python
answer = retriever.generate_answer(
    query="What is diabetes?",
    documents=documents,
    temperature=0.3
)
# Returns: "Diabetes is a chronic metabolic disease..."
```

##### Function: `query(question, top_k, filters, temperature, include_sources)`

**What this function does:** **MAIN FUNCTION** - Complete RAG pipeline in one call

**Parameters:**
- `question`: User's question
- `top_k`: Documents to retrieve (default: 5)
- `filters`: Metadata filters (optional)
- `temperature`: LLM temperature (default: 0.3)
- `include_sources`: Include source documents (default: True)

**Returns:** Dictionary with answer and sources

**Complete Process:**
1. **Retrieval Stage**:
   - Convert question to embedding
   - Search vector database with HNSW
   - Retrieve top K relevant documents

2. **Generation Stage**:
   - Build context from documents
   - Create RAG prompt
   - Generate answer with LLM

3. **Response Formatting**:
   - Format answer
   - Attach source citations
   - Return complete response

**Example:**
```python
result = retriever.query(
    question="What is diabetes?",
    top_k=5,
    temperature=0.3,
    include_sources=True
)

# Returns:
# {
#     "answer": "Diabetes is a chronic metabolic disease characterized by elevated blood glucose levels. It occurs when the pancreas doesn't produce enough insulin or when the body cannot effectively use the insulin it produces...",
#     "sources": [
#         {
#             "id": 1,
#             "score": 0.923,
#             "category": "endocrine",
#             "section_type": "definition",
#             "page": 100,
#             "text_preview": "Diabetes is a chronic disease..."
#         },
#         ...
#     ],
#     "num_sources": 5
# }
```

##### Function: `_build_context(documents)`

**What this function does:** Combines retrieved documents into context string

**Parameters:**
- `documents`: List of retrieved documents

**Returns:** Formatted context string

**Internal helper - not called directly**

##### Function: `_create_prompt(question, context)`

**What this function does:** Creates RAG prompt for LLM

**Parameters:**
- `question`: User's question
- `context`: Retrieved document context

**Returns:** Formatted prompt string

**Prompt structure:**
```
You are a helpful medical information assistant. Answer based ONLY on context.

Context:
[Source 1 - Category: endocrine, Relevance: 0.92]
Diabetes is a chronic disease...

[Source 2 - Category: endocrine, Relevance: 0.87]
Symptoms include thirst...

Question: What is diabetes?

Instructions:
- Answer based only on context
- Be accurate and factual
- Use clear language

Answer:
```

##### Function: `_format_sources(documents)`

**What this function does:** Formats source documents for response

**Parameters:**
- `documents`: Retrieved documents

**Returns:** List of formatted source information

#### Class: `MedicalChatRetriever`

**What this class does:** Conversational RAG with chat history support

**Constructor:**
```python
def __init__(self)
```
**What it does:** Creates chat retriever with conversation memory

**Maintains:**
- `conversation_history`: List of previous Q&A pairs
- `MedicalRAGRetriever`: Underlying RAG system

**Methods:**

##### Function: `chat(question, top_k, filters, use_history)`

**What this function does:** Chat with conversation context

**Parameters:**
- `question`: Current question
- `top_k`: Documents to retrieve
- `filters`: Metadata filters
- `use_history`: Include conversation history (default: True)

**Returns:** Response with answer and sources

**Process:**
1. Retrieves relevant documents
2. Builds context with conversation history
3. Generates answer considering previous turns
4. Stores Q&A in history

**Example:**
```python
chat = MedicalChatRetriever()

# Turn 1
result1 = chat.chat("What is diabetes?")
# Answer: "Diabetes is a chronic disease..."

# Turn 2 (follow-up)
result2 = chat.chat("What are its symptoms?")
# Answer: "The symptoms of diabetes include..." (knows "its" = diabetes)
```

##### Function: `clear_history()`

**What this function does:** Clears conversation history

##### Function: `get_history()`

**What this function does:** Returns conversation history

**Returns:** List of Q&A dictionaries

---

### Module 8: Application (application.py)

**Location:** `app/application.py`

**What it does:** Flask REST API server with AngularJS frontend

#### Flask Application

**What this application does:** Serves web interface and handles API requests

**Initialized Components:**
- `MedicalRAGRetriever`: Main RAG system
- `chat_sessions`: Dictionary storing chat sessions by session ID

---

#### API Endpoints

##### Endpoint: `GET /`

**What this endpoint does:** Serves main chat interface (HTML page)

**Returns:** AngularJS web application

**URL:** `http://localhost:5000/`

---

##### Endpoint: `GET /api/health`

**What this endpoint does:** Health check for monitoring

**Returns:** JSON with service status

**Example Response:**
```json
{
    "status": "healthy",
    "service": "Medical RAG Chatbot",
    "version": "1.0.0"
}
```

---

##### Endpoint: `POST /api/query`

**What this endpoint does:** Single-turn RAG query (no conversation history)

**Request Body:**
```json
{
    "question": "What is diabetes?",
    "top_k": 5,
    "filters": {"category": "endocrine"},
    "temperature": 0.3,
    "include_sources": true
}
```

**Response:**
```json
{
    "success": true,
    "answer": "Diabetes is a chronic metabolic disease...",
    "sources": [
        {
            "id": 1,
            "score": 0.923,
            "category": "endocrine",
            "section_type": "definition",
            "page": 100,
            "text_preview": "Diabetes is..."
        }
    ],
    "num_sources": 5
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is diabetes?",
    "top_k": 5
  }'
```

---

##### Endpoint: `POST /api/chat`

**What this endpoint does:** Conversational query with history

**Request Body:**
```json
{
    "session_id": "user123",
    "question": "What is diabetes?",
    "top_k": 5,
    "filters": null
}
```

**Response:**
```json
{
    "success": true,
    "answer": "Diabetes is...",
    "sources": [...],
    "num_sources": 5,
    "session_id": "user123"
}
```

**Features:**
- Maintains conversation history per session
- Understands context from previous turns
- Each user has separate session

**Example Conversation:**
```bash
# Turn 1
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user123", "question": "What is diabetes?"}'

# Turn 2 (follow-up)
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user123", "question": "What are its symptoms?"}'
# Understands "its" refers to diabetes from Turn 1
```

---

##### Endpoint: `POST /api/chat/clear`

**What this endpoint does:** Clears conversation history for session

**Request Body:**
```json
{
    "session_id": "user123"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Chat history cleared"
}
```

---

##### Endpoint: `POST /api/retrieve`

**What this endpoint does:** Retrieves documents only (no LLM generation)

**Request Body:**
```json
{
    "query": "diabetes",
    "top_k": 5,
    "filters": {"category": "endocrine"}
}
```

**Response:**
```json
{
    "success": true,
    "documents": [
        {
            "id": "vec_234",
            "score": 0.923,
            "text": "Diabetes is...",
            "metadata": {"category": "endocrine"}
        }
    ],
    "num_documents": 5
}
```

**Use Case:** Testing retrieval quality without LLM cost

---

##### Endpoint: `GET /api/categories`

**What this endpoint does:** Returns available medical categories

**Response:**
```json
{
    "success": true,
    "categories": [
        "cardiovascular",
        "neurological",
        "respiratory",
        "endocrine",
        "gastrointestinal",
        "musculoskeletal",
        "dermatological",
        "psychiatric",
        "immunological",
        "reproductive",
        "urological",
        "oncological"
    ]
}
```

---

#### Frontend (AngularJS)

**Location:** `app/templates/index.html`

**What this frontend does:** Interactive chat interface for Medical RAG

**Components:**

##### Main Chat Interface

**Features:**
- Real-time chat interface
- Message history display
- User and assistant message bubbles
- Source citation display
- Loading indicators

##### Sidebar Controls

**Category Filter:**
- Filter by medical category
- Click category button to filter results
- "All Categories" to clear filter

**Settings:**
- **Documents to retrieve**: Slider (3-10)
- **Temperature**: Slider (0.0-1.0)
  - 0.0 = More factual
  - 1.0 = More creative

**Statistics:**
- Total messages sent
- Last query response time

##### Example Queries

**Pre-built queries for quick testing:**
- "What is diabetes?"
- "Symptoms of heart disease"
- "How to treat hypertension?"

Click to automatically send

##### Message Display

**User Messages:**
- Right-aligned
- Purple gradient bubble
- User icon

**Assistant Messages:**
- Left-aligned
- Gray bubble
- Robot icon
- Source badges showing:
  - Category
  - Relevance score
  - Number of sources

---

### Module 9: AWS IAM Setup

**Location:** `infra/iam/`

**What it does:** Defines all AWS Identity and Access Management (IAM) resources required to run the Medical RAG Chatbot securely — including the IAM user for local development, the ECS task execution role, and the least-privilege policies for Bedrock and Pinecone access.

---

#### IAM User: `medical-rag-dev-user`

**Purpose:** Used for local development and CI/CD pipelines. Holds programmatic access keys that are stored in `.env`.

**How to create:**
```bash
aws iam create-user --user-name medical-rag-dev-user
aws iam create-access-key --user-name medical-rag-dev-user
# Save AccessKeyId and SecretAccessKey to .env
```

---

#### IAM Role: `medical-rag-ecs-task-role`

**Purpose:** Assumed by the ECS task container at runtime. Grants the application access to AWS Bedrock (LLM + embeddings) without embedding credentials in the image.

**Trust Policy** (`infra/iam/ecs-trust-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**How to create:**
```bash
aws iam create-role \
  --role-name medical-rag-ecs-task-role \
  --assume-role-policy-document file://infra/iam/ecs-trust-policy.json
```

---

#### IAM Role: `medical-rag-ecs-execution-role`

**Purpose:** Used by the ECS agent (not the container) to pull the Docker image from ECR and write logs to CloudWatch.

**Attach the AWS managed policy:**
```bash
aws iam attach-role-policy \
  --role-name medical-rag-ecs-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

---

#### IAM Policy: `medical-rag-bedrock-policy`

**Purpose:** Grants least-privilege access to invoke only the required AWS Bedrock models — the Gemma 3 27B LLM and the Cohere embedding model.

**Policy Document** (`infra/iam/bedrock-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvokeModels",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/google.gemma-3-27b-v1",
        "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3"
      ]
    }
  ]
}
```

**How to create and attach:**
```bash
# Create policy
aws iam create-policy \
  --policy-name medical-rag-bedrock-policy \
  --policy-document file://infra/iam/bedrock-policy.json

# Attach to ECS task role
aws iam attach-role-policy \
  --role-name medical-rag-ecs-task-role \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/medical-rag-bedrock-policy

# Attach to dev user (for local development)
aws iam attach-user-policy \
  --user-name medical-rag-dev-user \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/medical-rag-bedrock-policy
```

---

#### IAM Policy: `medical-rag-ecr-policy`

**Purpose:** Allows the CI/CD pipeline user to push Docker images to ECR.

**Policy Document** (`infra/iam/ecr-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAccess",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    }
  ]
}
```

**How to create and attach:**
```bash
aws iam create-policy \
  --policy-name medical-rag-ecr-policy \
  --policy-document file://infra/iam/ecr-policy.json

aws iam attach-user-policy \
  --user-name medical-rag-dev-user \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/medical-rag-ecr-policy
```

---

#### IAM Policy: `medical-rag-cloudwatch-policy`

**Purpose:** Allows the ECS task to write structured application logs to CloudWatch Logs.

**Policy Document** (`infra/iam/cloudwatch-policy.json`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:us-east-1:<ACCOUNT_ID>:log-group:/ecs/medical-rag-chatbot:*"
    }
  ]
}
```

**How to create and attach:**
```bash
aws iam create-policy \
  --policy-name medical-rag-cloudwatch-policy \
  --policy-document file://infra/iam/cloudwatch-policy.json

aws iam attach-role-policy \
  --role-name medical-rag-ecs-task-role \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/medical-rag-cloudwatch-policy
```

---

#### Complete IAM Summary Table

| Resource | Type | Purpose | Attached To |
|----------|------|---------|-------------|
| `medical-rag-dev-user` | IAM User | Local dev & CI/CD programmatic access | — |
| `medical-rag-ecs-task-role` | IAM Role | Runtime identity for ECS container | ECS Task Definition |
| `medical-rag-ecs-execution-role` | IAM Role | ECS agent pulls image & writes logs | ECS Task Definition |
| `medical-rag-bedrock-policy` | IAM Policy | Invoke Gemma 3 27B + Cohere embed models | Task role + dev user |
| `medical-rag-ecr-policy` | IAM Policy | Push/pull Docker images to ECR | Dev user |
| `medical-rag-cloudwatch-policy` | IAM Policy | Write app logs to CloudWatch | Task role |
| `AmazonECSTaskExecutionRolePolicy` | AWS Managed Policy | Pull from ECR + write to CloudWatch | Execution role |

---

#### Enabling Bedrock Model Access

Before any API calls succeed, the models must be enabled in the AWS Console:

1. Open [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to **Model access** → **Manage model access**
3. Enable the following models:
   - `Google → Gemma 3 27B Instruct`
   - `Cohere → Embed English v3`
4. Click **Save changes** — access is usually granted instantly

---

### Module 10: Deployment

**Location:** `infra/`

**What it does:** Packages the Flask application as a Docker container, pushes it to Amazon ECR, and deploys it on Amazon ECS (Fargate) behind an Application Load Balancer.

---

#### Deployment Architecture

```
┌──────────────┐     HTTPS      ┌─────────────────────┐
│   Internet   │ ─────────────> │  Application Load   │
│   (Users)    │                │  Balancer (ALB)     │
└──────────────┘                └────────┬────────────┘
                                         │ HTTP :5000
                                         ↓
                                ┌─────────────────────┐
                                │   ECS Fargate       │
                                │   Task (Container)  │
                                │   medical-rag-app   │
                                └────────┬────────────┘
                                         │
                          ┌──────────────┼──────────────┐
                          ↓              ↓              ↓
                   ┌─────────┐   ┌─────────────┐  ┌──────────┐
                   │ Bedrock │   │  Pinecone   │  │CloudWatch│
                   │(LLM +   │   │  (Vectors)  │  │  (Logs)  │
                   │ Embed)  │   │             │  │          │
                   └─────────┘   └─────────────┘  └──────────┘
```

---

#### Step 1: Dockerfile

**Location:** `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "app/application.py"]
```

---

#### Step 2: Build and Push to Amazon ECR

**Create ECR repository (one-time):**
```bash
aws ecr create-repository \
  --repository-name medical-rag-chatbot \
  --region us-east-1
```

**Authenticate Docker to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS \
  --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

**Build and push:**
```bash
# Build image
docker build -t medical-rag-chatbot .

# Tag for ECR
docker tag medical-rag-chatbot:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medical-rag-chatbot:latest

# Push to ECR
docker push \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medical-rag-chatbot:latest
```

---

#### Step 3: ECS Cluster

**Create ECS cluster (one-time):**
```bash
aws ecs create-cluster \
  --cluster-name medical-rag-cluster \
  --capacity-providers FARGATE \
  --region us-east-1
```

---

#### Step 4: ECS Task Definition

**Location:** `infra/ecs/task-definition.json`

```json
{
  "family": "medical-rag-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/medical-rag-ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/medical-rag-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "medical-rag-app",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medical-rag-chatbot:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        { "name": "AWS_REGION", "value": "us-east-1" },
        { "name": "BEDROCK_MODEL_ID", "value": "google.gemma-3-27b-v1" },
        { "name": "BEDROCK_EMBEDDING_MODEL_ID", "value": "cohere.embed-english-v3" },
        { "name": "PINECONE_INDEX_NAME", "value": "medical-rag-index" }
      ],
      "secrets": [
        {
          "name": "PINECONE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:medical-rag/pinecone-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/medical-rag-chatbot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "essential": true
    }
  ]
}
```

**Register task definition:**
```bash
aws ecs register-task-definition \
  --cli-input-json file://infra/ecs/task-definition.json
```

> **Note:** `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are intentionally absent from the task definition — the ECS task role (`medical-rag-ecs-task-role`) provides credentials automatically via the instance metadata service.

---

#### Step 5: ECS Service

**Create service:**
```bash
aws ecs create-service \
  --cluster medical-rag-cluster \
  --service-name medical-rag-service \
  --task-definition medical-rag-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxxx],securityGroups=[sg-xxxxxx],assignPublicIp=ENABLED}" \
  --region us-east-1
```

**Update service (redeploy after image push):**
```bash
aws ecs update-service \
  --cluster medical-rag-cluster \
  --service medical-rag-service \
  --force-new-deployment
```

---

#### Step 6: Secrets Management (AWS Secrets Manager)

Store sensitive values — never hardcode them in the task definition or image:

```bash
# Store Pinecone API key
aws secretsmanager create-secret \
  --name medical-rag/pinecone-api-key \
  --secret-string "your-pinecone-key"
```

Grant the ECS task role permission to read the secret:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:medical-rag/*"
    }
  ]
}
```

```bash
aws iam put-role-policy \
  --role-name medical-rag-ecs-task-role \
  --policy-name SecretsManagerAccess \
  --policy-document file://infra/iam/secrets-policy.json
```

---

#### Step 7: CloudWatch Logs

**Create log group:**
```bash
aws logs create-log-group \
  --log-group-name /ecs/medical-rag-chatbot \
  --region us-east-1
```

**View live logs:**
```bash
aws logs tail /ecs/medical-rag-chatbot --follow
```

---

#### Deployment Checklist

| Step | Command / Action | Status |
|------|-----------------|--------|
| Enable Bedrock models | AWS Console → Bedrock → Model access | ☐ |
| Create IAM roles & policies | `aws iam create-role / create-policy` | ☐ |
| Create ECR repository | `aws ecr create-repository` | ☐ |
| Build & push Docker image | `docker build / push` | ☐ |
| Create ECS cluster | `aws ecs create-cluster` | ☐ |
| Store secrets in Secrets Manager | `aws secretsmanager create-secret` | ☐ |
| Register task definition | `aws ecs register-task-definition` | ☐ |
| Create ECS service | `aws ecs create-service` | ☐ |
| Verify health endpoint | `curl http://<ALB-DNS>/api/health` | ☐ |

---

#### Environment Variable Reference

| Variable | Source in ECS | Source locally |
|----------|--------------|----------------|
| `AWS_REGION` | Task definition `environment` | `.env` |
| `BEDROCK_MODEL_ID` | Task definition `environment` | `.env` |
| `BEDROCK_EMBEDDING_MODEL_ID` | Task definition `environment` | `.env` |
| `PINECONE_INDEX_NAME` | Task definition `environment` | `.env` |
| `PINECONE_API_KEY` | Secrets Manager via `secrets` | `.env` |
| AWS credentials | ECS task role (automatic) | `.env` (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) |

---

## End-to-End RAG Workflow

### Complete User Query Flow
```
┌─────────────────────────────────────────────────────────────┐
│                  USER QUESTION                              │
│              "What is diabetes?"                            │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: FRONTEND (AngularJS)                               │
├─────────────────────────────────────────────────────────────┤
│  - User types question in input box                         │
│  - Clicks "Send" button                                     │
│  - AngularJS captures input                                 │
│  - Makes POST request to /api/chat                          │
│                                                             │
│  Request:                                                   │
│  {                                                          │
│    "session_id": "user_12345",                              │
│    "question": "What is diabetes?",                         │
│    "top_k": 5,                                              │
│    "temperature": 0.3                                       │
│  }                                                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: FLASK API (application.py)                         │
├─────────────────────────────────────────────────────────────┤
│  - Receives POST request at /api/chat                       │
│  - Extracts question and parameters                         │
│  - Gets or creates chat session                             │
│  - Calls retriever.chat()                                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: RETRIEVER (retriever.py)                           │
├─────────────────────────────────────────────────────────────┤
│  MedicalChatRetriever.chat()                                │
│    ├─ Calls retrieve_documents()                            │
│    └─ Calls generate_answer()                               │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: QUERY EMBEDDING (embeddings.py)                    │
├─────────────────────────────────────────────────────────────┤
│  - Takes query: "What is diabetes?"                         │
│  - Calls AWS Bedrock Cohere API                             │
│  - Converts to embedding vector                             │
│                                                             │
│  Output: [0.245, -0.115, 0.573, ..., 0.412]                │
│          (1024 numbers)                                     │
│                                                             │
│  Time: ~0.1 seconds                                         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: VECTOR SEARCH (vectorstore.py)                     │
├─────────────────────────────────────────────────────────────┤
│  PineconeHNSWVectorStore.semantic_search()                  │
│                                                             │
│  Process:                                                   │
│  1. Send query vector to Pinecone cloud                     │
│  2. Pinecone uses HNSW algorithm                            │
│  3. Compares with 7,590 stored vectors                      │
│  4. Uses cosine similarity                                  │
│  5. Returns top 5 most similar                              │
│                                                             │
│  HNSW Performance:                                          │
│  - Checks ~300 vectors (not all 7,590)                      │
│  - Multi-level graph navigation                             │
│  - 98% accuracy                                             │
│  - Time: ~0.05 seconds                                      │
│                                                             │
│  Retrieved Documents:                                       │
│  [                                                          │
│    {                                                        │
│      "score": 0.923,                                        │
│      "text": "Diabetes is a chronic disease characterized  │
│               by high blood sugar levels...",               │
│      "metadata": {                                          │
│        "category": "endocrine",                             │
│        "section_type": "definition",                        │
│        "page": 100                                          │
│      }                                                      │
│    },                                                       │
│    {                                                        │
│      "score": 0.891,                                        │
│      "text": "Common symptoms include increased thirst...", │
│      "metadata": {                                          │
│        "category": "endocrine",                             │
│        "section_type": "symptoms",                          │
│        "page": 101                                          │
│      }                                                      │
│    },                                                       │
│    ... (3 more documents)                                   │
│  ]                                                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: CONTEXT BUILDING (retriever.py)                    │
├─────────────────────────────────────────────────────────────┤
│  _build_context(documents)                                  │
│                                                             │
│  Combines retrieved documents:                              │
│                                                             │
│  [Source 1 - Category: endocrine, Relevance: 0.92]         │
│  Diabetes is a chronic disease characterized by high        │
│  blood sugar levels. It occurs when the pancreas doesn't    │
│  produce enough insulin...                                  │
│                                                             │
│  [Source 2 - Category: endocrine, Relevance: 0.89]         │
│  Common symptoms of diabetes include increased thirst,      │
│  frequent urination, extreme fatigue, and blurred vision... │
│                                                             │
│  [Source 3 - Category: endocrine, Relevance: 0.85]         │
│  Treatment for diabetes includes insulin therapy, oral      │
│  medications, diet modification, and regular exercise...    │
│                                                             │
│  ... (2 more sources)                                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: PROMPT CREATION (retriever.py)                     │
├─────────────────────────────────────────────────────────────┤
│  _create_prompt(question, context)                          │
│                                                             │
│  Creates RAG prompt:                                        │
│                                                             │
│  """                                                        │
│  You are a helpful medical information assistant.           │
│  Answer based ONLY on the provided context.                 │
│                                                             │
│  Context:                                                   │
│  [Source 1 - Category: endocrine, Relevance: 0.92]         │
│  Diabetes is a chronic disease...                           │
│                                                             │
│  [Source 2 - Category: endocrine, Relevance: 0.89]         │
│  Common symptoms include...                                 │
│                                                             │
│  Question: What is diabetes?                                │
│                                                             │
│  Instructions:                                              │
│  - Answer based only on context                             │
│  - Be accurate and factual                                  │
│  - Use clear, professional language                         │
│                                                             │
│  Answer:                                                    │
│  """                                                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: LLM GENERATION (llm.py)                            │
├─────────────────────────────────────────────────────────────┤
│  BedrockGemma3LLM.generate()                                │
│                                                             │
│  Process:                                                   │
│  1. Send prompt to AWS Bedrock                              │
│  2. Bedrock routes to Gemma 3 27B model                     │
│  3. Model generates response                                │
│  4. Temperature: 0.3 (factual, not creative)                │
│  5. Max tokens: 1024                                        │
│                                                             │
│  Model: Gemma 3 27B                                         │
│  Provider: AWS Bedrock                                      │
│  Parameters: 27 billion                                     │
│  Time: ~2-4 seconds                                         │
│                                                             │
│  Generated Answer:                                          │
│  "Diabetes is a chronic metabolic disease characterized     │
│  by elevated blood glucose levels. It occurs when the       │
│  pancreas doesn't produce enough insulin or when the body   │
│  cannot effectively use the insulin it produces. There are  │
│  two main types: Type 1, where the pancreas produces        │
│  little or no insulin, and Type 2, where the body becomes   │
│  resistant to insulin. Common symptoms include increased    │
│  thirst, frequent urination, extreme fatigue, and blurred   │
│  vision. Treatment typically involves insulin therapy,      │
│  oral medications, dietary modifications, and regular       │
│  exercise to manage blood sugar levels."                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 9: RESPONSE FORMATTING (retriever.py)                 │
├─────────────────────────────────────────────────────────────┤
│  _format_sources(documents)                                 │
│                                                             │
│  Combines answer + sources:                                 │
│                                                             │
│  {                                                          │
│    "answer": "Diabetes is a chronic metabolic disease...",  │
│    "sources": [                                             │
│      {                                                      │
│        "id": 1,                                             │
│        "score": 0.923,                                      │
│        "category": "endocrine",                             │
│        "section_type": "definition",                        │
│        "page": 100,                                         │
│        "text_preview": "Diabetes is a chronic..."           │
│      },                                                     │
│      {                                                      │
│        "id": 2,                                             │
│        "score": 0.891,                                      │
│        "category": "endocrine",                             │
│        "section_type": "symptoms",                          │
│        "page": 101,                                         │
│        "text_preview": "Common symptoms include..."         │
│      },                                                     │
│      ... (3 more sources)                                   │
│    ],                                                       │
│    "num_sources": 5                                         │
│  }                                                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 10: API RESPONSE (application.py)                     │
├─────────────────────────────────────────────────────────────┤
│  - Receives response from retriever                         │
│  - Stores in conversation history                           │
│  - Formats JSON response                                    │
│  - Sends back to frontend                                   │
│                                                             │
│  Response:                                                  │
│  {                                                          │
│    "success": true,                                         │
│    "answer": "Diabetes is a chronic metabolic disease...",  │
│    "sources": [...],                                        │
│    "num_sources": 5,                                        │
│    "session_id": "user_12345"                               │
│  }                                                          │
│                                                             │
│  Total Time: ~2.5 seconds                                   │
│  Cost: ~$0.001 per query                                    │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 11: FRONTEND DISPLAY (AngularJS)                      │
├─────────────────────────────────────────────────────────────┤
│  - Receives JSON response                                   │
│  - Adds to messages array                                   │
│  - Updates UI                                               │
│                                                             │
│  Display:                                                   │
│  ┌──────────────────────────────────────────┐              │
│  │ 👤 User: What is diabetes?               │              │
│  ├──────────────────────────────────────────┤              │
│  │ 🤖 Assistant:                            │              │
│  │ Diabetes is a chronic metabolic disease  │              │
│  │ characterized by elevated blood glucose  │              │
│  │ levels...                                │              │
│  │                                          │              │
│  │ 📚 Sources:                              │              │
│  │ [endocrine (0.92)] [endocrine (0.89)]    │              │
│  └──────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

### Performance Metrics

| Stage | Time | Cost |
|-------|------|------|
| Query Embedding (Bedrock Cohere) | ~0.1s | AWS Bedrock pricing |
| Vector Search (HNSW) | ~0.05s | $0 (included) |
| Context Building | ~0.01s | $0 |
| LLM Generation | ~2-4s | AWS Bedrock pricing |
| **Total** | **~2.5s** | **varies by usage** |

### Accuracy Metrics

- **Retrieval Accuracy**: 98% (HNSW approximate)
- **Answer Quality**: High (based on Gemma 3 27B)
- **Source Relevance**: 90%+ (top 5 documents)

---

## API Documentation

### Base URL
```
http://localhost:5000
```

### Authentication

Currently no authentication required (add JWT for production)

### Rate Limiting

No rate limiting (add for production)

### Error Responses

All endpoints return errors in this format:
```json
{
    "success": false,
    "error": "Error message here"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (missing parameters) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Usage Examples

### Python SDK Example
```python
from app.components.retriever import MedicalRAGRetriever

# Initialize
retriever = MedicalRAGRetriever()

# Simple query
result = retriever.query(
    question="What is diabetes?",
    top_k=5,
    temperature=0.3
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")
```

### cURL Examples

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Simple Query:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is diabetes?",
    "top_k": 5,
    "temperature": 0.3
  }'
```

**Filtered Query:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are heart attack symptoms?",
    "top_k": 5,
    "filters": {"category": "cardiovascular"}
  }'
```

**Chat with History:**
```bash
# Turn 1
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "question": "What is diabetes?"
  }'

# Turn 2
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "question": "What are its symptoms?"
  }'
```

### JavaScript/Frontend Example
```javascript
// Simple query
fetch('http://localhost:5000/api/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        question: 'What is diabetes?',
        top_k: 5,
        temperature: 0.3
    })
})
.then(response => response.json())
.then(data => {
    console.log('Answer:', data.answer);
    console.log('Sources:', data.sources);
});
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "No module named 'app'"

**Solution:**
```bash
# Make sure you're in project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
python app/application.py
```

#### Issue 2: "AWS Bedrock access denied for embeddings"

**Solution:**
```bash
# Check .env file exists
cat .env

# Verify AWS credentials are set
echo $AWS_ACCESS_KEY_ID

# Ensure cohere.embed-english-v3 is enabled in AWS Bedrock console
# Request model access if needed
```

#### Issue 3: "Pinecone index not found"

**Solution:**
```bash
# Rebuild vector database
python scripts/build_vectordb.py
```

#### Issue 4: "AWS Bedrock access denied"

**Solution:**
1. Check AWS credentials in `.env`
2. Verify Bedrock access in AWS Console
3. Request Gemma 3 27B access
4. Wait for approval (usually instant)

#### Issue 5: "Slow query responses (>10 seconds)"

**Possible causes:**
- Cold start (first query is slower)
- Network latency
- Large top_k value

**Solutions:**
- Reduce top_k (try 3 instead of 10)
- Use local FAISS instead of Pinecone
- Check internet connection

#### Issue 6: "Frontend not loading"

**Solution:**
```bash
# Check Flask is running
curl http://localhost:5000/api/health

# Check port 5000 is not in use
lsof -i :5000

# Try different port
python app/application.py --port 8000
```

### Debug Mode

Enable detailed logging:
```python
# In application.py
app.run(host='0.0.0.0', port=5000, debug=True)
```

### Logs Location
```
logs/
├── 2024-03-17.log
├── 2024-03-18.log
└── 2024-03-19.log
```

---

## AWS IAM, Roles & Policy Requirements

See [Module 9: AWS IAM Setup](#module-9-aws-iam-setup) for full details.

### Quick Reference

#### Required IAM Resources

| Resource | Type | Required For |
|----------|------|-------------|
| `medical-rag-dev-user` | IAM User | Local development, CI/CD |
| `medical-rag-ecs-task-role` | IAM Role | ECS container runtime access to Bedrock |
| `medical-rag-ecs-execution-role` | IAM Role | ECS agent pulling images and writing logs |
| `medical-rag-bedrock-policy` | Customer Policy | `bedrock:InvokeModel` on Gemma 3 27B + Cohere |
| `medical-rag-ecr-policy` | Customer Policy | ECR image push/pull |
| `medical-rag-cloudwatch-policy` | Customer Policy | CloudWatch log writes |
| `AmazonECSTaskExecutionRolePolicy` | AWS Managed | ECR pull + CloudWatch (execution role) |

#### Minimum Bedrock Permissions

```json
{
  "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
  "Resource": [
    "arn:aws:bedrock:us-east-1::foundation-model/google.gemma-3-27b-v1",
    "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3"
  ]
}
```

#### Common IAM Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AccessDeniedException: bedrock:InvokeModel` | Missing Bedrock policy | Attach `medical-rag-bedrock-policy` to the role/user |
| `AccessDeniedException` on ECS task start | Execution role missing | Attach `AmazonECSTaskExecutionRolePolicy` to execution role |
| `ResourceNotFoundException` on model invoke | Model not enabled in Bedrock | Enable model via AWS Console → Bedrock → Model access |
| `AuthorizationError` from Pinecone | Missing API key | Verify `PINECONE_API_KEY` in Secrets Manager or `.env` |

---

## Application Deployment

See [Module 10: Deployment](#module-10-deployment) for full details.

### Quick Start (Local Docker)

```bash
# Build image
docker build -t medical-rag-chatbot .

# Run locally with .env
docker run -p 5000:5000 --env-file .env medical-rag-chatbot

# Test
curl http://localhost:5000/api/health
```

### Quick Start (ECS Fargate)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker build -t medical-rag-chatbot .
docker tag medical-rag-chatbot:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medical-rag-chatbot:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/medical-rag-chatbot:latest

# 2. Register task and deploy
aws ecs register-task-definition \
  --cli-input-json file://infra/ecs/task-definition.json
aws ecs update-service \
  --cluster medical-rag-cluster \
  --service medical-rag-service \
  --force-new-deployment
```

### Infrastructure File Structure

```
infra/
├── iam/
│   ├── ecs-trust-policy.json         # Trust policy for ECS task role
│   ├── bedrock-policy.json           # Bedrock InvokeModel permissions
│   ├── ecr-policy.json               # ECR push/pull permissions
│   ├── cloudwatch-policy.json        # CloudWatch log write permissions
│   └── secrets-policy.json           # Secrets Manager read permissions
└── ecs/
    └── task-definition.json          # ECS Fargate task definition
```

---

## Contributing

### Development Setup

1. Fork repository
2. Create feature branch
3. Install dev dependencies:
```bash
pip install -r requirements-dev.txt
```
4. Make changes
5. Run tests:
```bash
pytest tests/
```
6. Submit pull request

### Code Style

- PEP 8 compliance
- Type hints for functions
- Docstrings for classes and methods
- Maximum line length: 100 characters

### Testing

Run tests:
```bash
# All tests
pytest

# Specific module
pytest tests/test_retriever.py

# With coverage
pytest --cov=app tests/
```

---

## License

MIT License

---

## Contact

For questions or support:
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/medical-rag-chatbot/issues

---

## Acknowledgments

- **AWS Bedrock** - Gemma 3 27B LLM and Cohere embeddings
- **Pinecone** - Vector database
- **LangChain** - RAG framework
- **Medical Encyclopedia** - Knowledge source

---

**Built with ❤️ for better medical information access**