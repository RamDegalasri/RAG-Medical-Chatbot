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

    # AWS Bedrock Embedding Model
    AWS_BEDROCK_EMBEDDING_MODEL = "cohere.embed-english-v3"

    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "medical-rag-index-cohere"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    # AWS Bedrock Confirguation
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")

    # Bedrock model ID
    BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

    # Data Paths
    DATAPATH = "data/"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50