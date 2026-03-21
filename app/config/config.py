import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Confirguation
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"
    
    # OpenAI Model and Embedding key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "medical-rag-index"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    # Data Paths
    DATAPATH = "data/"
    DB_FAISS_PATH = "vectorstore/DBpinecone"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50