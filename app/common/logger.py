# app/common/logger.py

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file with date
LOG_FILE = os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# Error log file
ERROR_LOG_FILE = os.path.join(LOGS_DIR, "error.log")


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


# File formatter (without colors)
file_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(CustomFormatter())

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Error file handler
error_file_handler = RotatingFileHandler(
    ERROR_LOG_FILE,
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding='utf-8'
)
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(file_formatter)


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure logger
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class MedicalRAGLogger:
    """Logger wrapper with Medical RAG specific methods"""
    
    def __init__(self, name: str = __name__):
        self.logger = get_logger(name)
    
    def log_query(self, query: str, user_id: str = None):
        """Log user query"""
        self.logger.info(f"USER_QUERY | User: {user_id} | Query: {query}")
    
    def log_retrieval(self, num_docs: int, query: str):
        """Log document retrieval"""
        self.logger.info(f"RETRIEVAL | Retrieved {num_docs} documents for query: {query[:50]}...")
    
    def log_response(self, response: str, query: str):
        """Log generated response"""
        self.logger.info(f"RESPONSE | Query: {query[:50]}... | Response length: {len(response)} chars")
    
    def log_pdf_processing(self, filename: str, num_pages: int):
        """Log PDF processing"""
        self.logger.info(f"PDF_PROCESS | File: {filename} | Pages: {num_pages}")
    
    def log_embedding(self, num_chunks: int, model: str):
        """Log embedding generation"""
        self.logger.info(f"EMBEDDING | Chunks: {num_chunks} | Model: {model}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        self.logger.error(f"ERROR | Context: {context} | Error: {str(error)}", exc_info=True)
    
    def log_api_call(self, endpoint: str, status_code: int, response_time: float):
        """Log API calls"""
        self.logger.info(f"API_CALL | Endpoint: {endpoint} | Status: {status_code} | Time: {response_time:.2f}s")
    
    def log_model_performance(self, latency: float, tokens: int = None):
        """Log model performance metrics"""
        self.logger.info(f"MODEL_PERF | Latency: {latency:.2f}s | Tokens: {tokens}")
    
    def log_vectorstore_operation(self, operation: str, collection_name: str, status: str):
        """Log vector store operations"""
        self.logger.info(f"VECTORSTORE | Operation: {operation} | Collection: {collection_name} | Status: {status}")


# Test
if __name__ == "__main__":
    test_logger = MedicalRAGLogger("test")
    
    test_logger.logger.debug("Debug message")
    test_logger.logger.info("Info message")
    test_logger.logger.warning("Warning message")
    test_logger.logger.error("Error message")
    
    test_logger.log_query("What is diabetes?", "user123")
    test_logger.log_pdf_processing("medical.pdf", 150)
    
    print(f"\nLogs saved to: {LOG_FILE}")