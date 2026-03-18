import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATAPATH = "data/"
    DB_FAISS_PATH = ""
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50