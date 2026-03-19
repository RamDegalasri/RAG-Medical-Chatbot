from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.config.config import Config
from app.common.logger import MedicalRAGLogger

from typing import List, Dict, Tuple
import numpy as np

class Medical_Embeddings:
    """
    Generate embeddings for medical documents using OpenAI
    This class focuses on embedding generation not vector storage
    """

    def __init__(self, model_name: str = Config.OPENAI_EMBEDDING_MODEL):
        """
        Initialize OpenAI embeddings model

        Args:
            model_name: OpenAI embedding model
                - "text-embedding-3-small" (1536 dims, recommended)
                - "text-embedding-3-large" (3072 dims, higher quality)
        """
        self.logger = MedicalRAGLogger(__name__)
        self.model_name = model_name

        # Initialize OpenAI embeddings
        self.logger.logger.info(f"Initializing OpenAI embedding model: {model_name}")

        try:
            self.embeddings = OpenAIEmbeddings(
                model = model_name,
                openai_api_key = Config.OPENAI_API_KEY
            )

            self.logger.logger.info("OpenAI embedding model loaded successfully")

        except Exception as e:
            self.logger.log_error(e, context = "Initializing OpenAI embeddings")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model

        Returns:
            Embedding dimension (1536 or 3072)
        """
        if "large" in self.model_name:
            return 3072
        else:
            return 1536
        
    def embed_single_chunk(self, chunk: Document) -> Tuple[List[float], Dict]:
        """
        Generate embedding for a single document chunk

        Args:
            chunk: Document chunk with page_content and metadata

        Returns:
            Tuple of embedding vector, metadata
        """
        try:
            # Generate embedding for the text
            embedding = self.embeddings.embed_query(chunk.page_content)
            return embedding, chunk.metadata
        
        except Exception as e:
            self.logger.log_error(e, context = f"Embedding single chunk: {chunk.page_content[:50]}...")
            raise

    def embed_chunks(self, chunks: List[Document]) -> List[Tuple[List[float], Dict]]:
        """
        Generate embeddings for multiple document chunks

        Args:
            chunks: List of Document chunks

        Returns:
            List of tuples embedding vector, metadata
        """

        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")
            
            self.logger.logger.info(f"Generating embeddings for {len(chunks)} chunks...")

            # Estimate cost
            cost_info = self.estimate_embedding_cost(chunks)
            self.logger.logger.info(
                f"Estimated cost: {cost_info['estimated_total_cost']}"
                f"({cost_info['estimated_tokens']} tokens)"
            )

            # Generate embeddings for all chunks
            embedded_chunks = []

            for i, chunk in enumerate(chunks):
                if (i + 1) % 100 == 0:
                    self.logger.logger.info(f"Progress: {i + 1}/{len(chunks)} chunks embedded")

                embedding, metadata = self.embed_single_chunk(chunk)
                embedded_chunks.append((embedding, metadata))

            self.logger.log_embedding(len(embedded_chunks), self.model_name)
            self.logger.logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")

            return embedded_chunks
        except Exception as e:
            self.logger.log_error(e, context = "Embedding multiple chunks")
            raise

    def embed_chunks_with_documents(self, chunks: List[Document]) -> List[Dict]:
        """
        Generate embeddings and return as structured dictionary

        Args:
            chunks: List of Document chunks

        Returns:
            List of dictionaries with structure:
            {
                'id': unique_id,
                'text': original_text,
                'embedding': vector,
                'metadata': metadata_dict
            }
        """
        try:
            self.logger.logger.info(f"Embedding {len(chunks)} chunks with full document structure...")

            embedded_documents = []

            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk.page_content)

                # Create structured document
                embedded_doc = {
                    'id': f"chunk_{i}",
                    'text': chunk.page_content,
                    'embedding': embedding,
                    'metadata': chunk.metadata
                } 

                embedded_documents.append(embedded_doc)

                if (i + 1) % 100 == 0:
                    self.logger.logger.info(f"Progress: {i + 1}/{len(chunks)}")

            self.logger.logger.info(f"Created {len(embedded_documents)} embedded documents")

            return embedded_documents
        
        except Exception as e:
            self.logger.log_error(e, context = "Embedding chunks with documents")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        try:
            self.logger.logger.info(f"Embedding query: '{query[:50]}...'")
            embedding = self.embeddings.embed_query(query)
            self.logger.logger.info(f"Query embedded successfully")
            return embedding
        
        except Exception as e:
            self.logger.log_error(e, context = f"Embedding query: {query}")
            raise

    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict:
        """
        Get statistics about generated embeddings

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary with statistics
        """
        try:
            embeddings_array = np.array(embeddings)

            stats = {
                "total_embeddings": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "avg_magnitude": float(np.mean(np.linalg.norm(embeddings_array, axis =1))),
                "std_magnitude": float(np.std(np.linalg.norm(embeddings_array, axis = 1))),
                "model": self.model_name        
            }

            return stats
        
        except Exception as e:
            self.logger.log_error(e, context = "Getting embedding stats")
            return {"error": str(e)}
        
    def estimate_embedding_cost(self, chunks: List[Document]) -> Dict:
        """
        Estimate cost for embedding documents

        Args:
            chunks: List of Document chunks

        Returns:
            Dictionary with cost estimates
        """

        # Count tokens (rough estimation: 1 token = 0.75 words)
        total_words = sum(len(chunk.page_content.split()) for chunk in chunks)
        estimated_tokens = int(total_words / 0.75)

        # OpenAI pricing (as of 2024)
        pricing = {
            "text-embedding-3-small": 0.00002,  # per 1K tokens
            "text-embedding-3-large": 0.00013,  # per 1K tokens
        }

        cost_per_1k = pricing.get(self.model_name, 0.00002)
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k

        return {
            "total_chunks": len(chunks),
            "total_words": total_words,
            "estimated_tokens": estimated_tokens,
            "model": self.model_name,
            "cost_per_1k_tokens": f"${cost_per_1k: .5f}",
            "estimated_total_cost": f"${estimated_cost: .4f}"
        }
    
class MedicalEmbeddingPipeline:
    """
    Complete pipeline for processing medical documents and generating embeddings.
    Integrates with pdf_loader to create end-to-end embedding generation.
    """

    def __init__(self, model_name: str = Config.OPENAI_EMBEDDING_MODEL):
        """
        Initialize embedding pipeline

        Args:
            model_name: OpenAI embedding model to use
        """
        self.logger = MedicalRAGLogger(__name__)
        self.embeddings_handler = Medical_Embeddings(model_name)

    def process_chunks_to_embeddings(self, chunks: List[Document],include_metadata: bool = True) -> List[Tuple[List[float], Dict]]:
        """
        Process document chunks and generate embeddings

        Args:
            chunks: List of Document chunks from pdf_loader
            include_metadata: Whether to include metadata in output

        Returns:
            List of (embeddings, metadata) tuples
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("STARTING EMBEDDING PIPELINE")
            self.logger.logger.info("="*60)

            # Step 1: Validate input
            if not chunks:
                raise ValueError("No chunks provided")
            
            self.logger.logger.info(f"Input: {len(chunks)} chunks")

            # Step 2: Show cost estimate
            cost_estimate = self.embeddings_handler.estimate_embedding_cost(chunks)
            self.logger.logger.info(f"\nCost Estimate:")
            for key, value in cost_estimate.items():
                self.logger.logger.info(f" {key}: {value}")

            # Step 3: Generate embeddings
            self.logger.logger.info(f"\nGenerating embeddings...")
            embedded_chunks = self.embeddings_handler.embed_chunks(chunks)

            # Step 4: Get embedding statistics
            embeddings_only = [emb for emb, _ in embedded_chunks]
            stats = self.embeddings_handler.get_embedding_stats(embeddings_only)

            self.logger.logger.info(f"\nEmbedding Statistics:")
            for key, value in stats.items():
                self.logger.logger.info(f" {key}: {value}")

            self.logger.logger.info("\n" + "=" * 60)
            self.logger.logger.info("EMBEDDING PIPELINE COMPLETED")
            self.logger.logger.info("=" * 60)

            return embedded_chunks
        
        except Exception as e:
            self.logger.log_error(e, context = "embedding pipeline")
            raise

    def process_chunks_to_documents(self, chunks: List[Document]) -> List[Dict]:
        """
        Process chunks and return structured documents with embeddings

        Args:
            chunks: List of Document chunks from pdf_loader

        Returns:
            List of dictionaries with full document structure
        """
        try:
            self.logger.logger.info("Processing chunks to structured documents.")

            # Show cost estimate
            cost_estimate = self.embeddings_handler.estimate_embedding_cost(chunks)
            self.logger.logger.info(f"Cost Estimate: {cost_estimate}")

            embedded_documents = self.embeddings_handler.embed_chunks_with_documents(chunks)
            self.logger.logger.info(f"Created {len(embedded_documents)} embedded documents")
            return embedded_documents
        
        except Exception as e:
            self.logger.log_error(e, context = "Processing chunks to documents")
            raise

    def get_embedding_for_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        try:
            return self.embeddings_handler.embed_query(query)
        except Exception as e:
            self.logger.log_error(e, context = "Getting query embedding")
            raise

# Utility function for easy integration with pdf_loader
def create_embeddings_from_chunks(chunks: List[Document], model_name: str = Config.OPENAI_EMBEDDING_MODEL) -> List[Tuple[List[float], Dict]]:
        """
        Convenience function to create embeddings from pdf_loader chunks

        Args:
            chunks: Document chunks from pdf_loader
            model_name: OpenAI embedding model

        Returns:
            List of (embedding, metadata) tuples
        """
        pipeline = MedicalEmbeddingPipeline(model_name)
        return pipeline.process_chunks_to_embeddings(chunks)

# Usage/Testing
if __name__ == "__main__":
    from app.components.pdf_loader import MedicalPDFLoader

    print("\n" + "=" * 60)
    print("MEDICAL EMBEDDING PIPELINE - TESTING")
    print("=" * 60 + "\n")

    # Step 1: Load and process PDF
    print("Step 1: Loading PDF and creating chunks...")
    print("-" * 60)

    try:
        # Load PDF (adjust filename as needed)
        pdf_loader = MedicalPDFLoader()
        chunks = pdf_loader.process_all_pdfs()

        # Use only first 10 chunks for testing
        test_chunks = chunks[:10]

        print(f"Loaded {len(test_chunks)} chunks for testing\n")

    except Exception as e:
        print(f"x Error loading PDF: {e}\n")
        print("Creating sample chunks for testing...\n")

        # Create sample chunks if PDF not available
        test_chunks = [
            Document(
                page_content="Diabetes is a chronic disease characterized by high blood sugar levels.",
                metadata={"category": "endocrine", "section_type": "definition", "page": 100}
            ),
            Document(
                page_content="Symptoms of diabetes include increased thirst, frequent urination, and fatigue.",
                metadata={"category": "endocrine", "section_type": "symptoms", "page": 101}
            ),
            Document(
                page_content="Treatment for diabetes includes insulin therapy and lifestyle modifications.",
                metadata={"category": "endocrine", "section_type": "treatment", "page": 102}
            ),
        ]
    
    # Step 2: Initialize embedding pipeline
    print("Step 2: Initializing embedding pipeline...")
    print("-" * 60)

    pipeline = MedicalEmbeddingPipeline(model_name = Config.OPENAI_EMBEDDING_MODEL)
    print("Pipeline initialized\n")

    # Step 3: Generate embeddings
    print("Ste3: Generating embeddings...")
    print("-" * 60)

    try:
        embedded_chunks = pipeline.process_chunks_to_documents(test_chunks)

        print(f"\n Successfully generated {len(embedded_chunks)} embeddings\n")

        # Show sample results
        print("Sample Results: ")
        print("-" * 60 )

        for i, (embedding, metadata) in enumerate(embedded_chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print(f"  Metadata: {metadata}")

    except Exception as e:
        print(f"x Error generating embeddings: {e}\n")

    # Step 4: Test structures document output
    print("\n" + "=" * 60)
    print("Step 4: Testing structured document output...")
    print("-" * 60)

    try:
        embedded_docs = pipeline.process_chunks_to_documents(test_chunks[:3])

        print(f"\n Created {len(embedded_docs)} structured documents\n")

        # Show structure
        print("Sample Document Strucutre:")
        print("-" * 60)
        sample_doc = embedded_docs[0]
        print(f"Keys: {list(sample_doc.keys())}")
        print(f"ID: {sample_doc['id']}")
        print(f"Text: {sample_doc['text'][:80]}...")
        print(f"Embedding dimension: {len(sample_doc['embedding'])}")
        print(f"Metadata: {sample_doc['metadata']}")
        
    except Exception as e:
        print(f"✗ Error creating structured documents: {e}\n")
    
    # Step 5: Test query embedding
    print("\n" + "="*60)
    print("Step 5: Testing query embedding...")
    print("-" * 60)
    
    try:
        query = "What is diabetes?"
        query_embedding = pipeline.get_embedding_for_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"✓ Generated query embedding")
        print(f"  Dimension: {len(query_embedding)}")
        print(f"  First 5 values: {query_embedding[:5]}")
        
    except Exception as e:
        print(f"✗ Error generating query embedding: {e}\n")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60 + "\n")
    
    print("Next Steps:")
    print("  1. Embeddings are ready for vector database storage")
    print("  2. Create vectorstore.py to handle Pinecone/FAISS operations")
    print("  3. Implement similarity search and retrieval")



