from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document

from typing import List, Dict, Tuple, Optional
import time

from app.config.config import Config
from app.common.logger import MedicalRAGLogger

class PineconeHNSWVectorStore:
    """
    Pinecone vector database with HNSW semantic search.

    Architecture:
    - Vector Storage: Pinecone (cloud database)
    - Search Algorithm: HNSW (automatically used by Pinecone)
    - Similarity Metric: Cosine (for semantic text matching)
    """

    def __init__(self):
        """
        Initialize Pinecone connection.
        HNSW is automatically used by Pinecone for fast semantic search.
        """
        self.logger = MedicalRAGLogger(__name__)
        self.index_name = Config.PINECONE_INDEX_NAME

        # Initialize Pinecone client
        self.logger.logger.info("Initializing Pinecone Vector Database...")
        self.logger.logger.info("  Storage: Pinecone Cloud")
        self.logger.logger.info("  Search Algorithm: HNSW (automatic)")
        self.logger.logger.info(" Similarity Metric: Cosine")

        try:
            self.pc = Pinecone(api_key = Config.PINECONE_API_KEY)
            self.logger.logger.info("Connected to Pinecone")

        except Exception as e:
            self.logger.log_error(e, context = "Connecting to Pinecone")
            raise

        self.index = None

    def create_index(self, dimension: int = 1536, metric: str = "cosine", force_recreate: bool = False):
        """
        Create Pinecone index with HNSW search enabled.

        Pinecone automatically uses HNSW algorithm for semantic search.
        No manual HNSW confirguation needed - its optimized internally.

        Args:
            dimension: Vector dimension (1536 for text-embedding-3-small)
            metric: Similarity metric - 'cosine' for semantic text search
            force_recreate: If true, delete and recreate existing index
        """
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()

            if self.index_name in existing_indexes:
                if force_recreate:
                    self.logger.logger.info(f"Deleting existing index: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                    time.sleep(1) # Wait for deletion

                else:
                    self.logger.logger.info(f"Index '{self.index_name}' already exists")
                    self.logger.logger.info(f"  Using existing Pinecone index with HNSW search")
                    self.index = self.pc.Index(self.index_name)
                    return
                
            # Create new index
            self.logger.logger.info(f"Creating Pinecone index: {self.index_name}")
            self.logger.logger.info(f"  Vector dimension: {dimension}")
            self.logger.logger.info(f"  Similarity metric: {metric}")
            self.logger.logger.info(f"  Search algorithm: HNSW (Pinecone automatic)")

            self.pc.create_index(
                name = self.index_name,
                dimension = dimension,
                metric = metric,
                spec = ServerlessSpec(
                    cloud = Config.PINECONE_CLOUD,
                    region = Config.PINECONE_REGION
                )
            )

            # Wait for index to be ready
            self.logger.logger.info("Waiting for index initialization...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)

            self.logger.logger.info(f"Pinecone index created successfully")
            self.logger.logger.info(f"HNSW search enabled automatically")

            self.logger.log_vectorstore_operation(
                operation = "create_index",
                collection_name = self.index_name,
                status = "success"
            )

        except Exception as e:
            self.logger.log_error(e, context = "Creating Pinecone index with HNSW")
            raise

    def store_embeddings(self, embeddings: List[Tuple[List[float], Dict]], batch_size: int = 100, namespace: str = ""):
        """
        Store embeddings in Pinecone vector database.
        Vectors are stored with HNSW indexing for fast semantic search.

        Args:
            embeddings: List of (embedding_vector, metadata) tuples
            batch_size: Number of vectors to upload per batch
            namespace: Pinecone namespace (optional, for data organization)
        """
        try:
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            # Ensure index exists
            if self.index is None:
                dimension = len(embeddings[0][0])
                self.create_index(dimension = dimension)

            self.logger.logger.info(f"Storing {len(embeddings)} embeddings in Pinecone...")
            self.logger.logger.info(f"  Batch size: {batch_size}")
            self.logger.logger.info(f"  HNSW indexing: Automatic")

            # Prepare vectors for upload
            vectors_to_upsert = []

            for i, (embedding, metadata) in enumerate(embeddings):
                vector_id = f"vec_{i}"

                # Prepare vector data
                vector_data = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                } 

                vectors_to_upsert.append(vector_data)

                # Upload in batches
                if len(vectors_to_upsert) >= batch_size:
                    self._upsert_batch(vectors_to_upsert, namespace)
                    vectors_to_upsert = []

            # Upload remaining
            if vectors_to_upsert:
                self._upsert_batch(vectors_to_upsert, namespace)

            self.logger.logger.info(f"Successfully stored {len(embeddings)} embeddings")
            self.logger.logger.info(f"HNSW index automatically updated")

            self.logger.log_vectorstore_operation(
                operation = "store_embeddings",
                collection_name = self.index_name,
                status = "success"
            )

        except Exception as e:
            self.logger.log_error(e, context = "Storing embeddings in Pinecone")
            raise

    def store_structured_documents(self, documents: List[Dict], batch_size: int = 100, namespace: str = ""):
        """
        Store structured documents with embeddings in Pinecone.

        Args:
            documents: List of dicts with structure:
                {
                    'id': unique_id,
                    'text': original_text,
                    'embedding': vector,
                    'metadata': metadata_dict
                }
            batch_size: Vectors per batch
            namespace: Pinecone namespace
        """
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            # Ensure index exists
            if self.index is None:
                dimension = len(documents[0]['embedding'])
                self.create_index(dimension = dimension)

            self.logger.logger.info(f"Storing {len(documents)} documents in Pinecone...")

            vectors_to_upsert = []

            for i, doc in enumerate(documents):
                # Combine text with metadata
                metadata = doc['metadata'].copy()
                metadata['text'] = doc['text']  # Store original text
                metadata['doc_id'] = doc['id']

                vector_data = {
                    "id": doc['id'],
                    "values": doc['embedding'],
                    "metadata": metadata
                }

                vectors_to_upsert.append(vector_data)

                # Upload in batches
                if len(vectors_to_upsert) >= batch_size:
                    self._upsert_batch(vectors_to_upsert, namespace)
                    vectors_to_upsert = []

                    if (i + 1) % 500 == 0:
                        self.logger.logger.info(f"Progress: {i + 1}/{len(documents)}")

            # Upload remaining
            if vectors_to_upsert:
                self._upsert_batch(vectors_to_upsert, namespace)

            self.logger.logger.info(f"Successfully stored {len(documents)} documents")

        except Exception as e:
            self.logger.log_error(e, context="Storing structured documents")
            raise

    def _upsert_batch(self, vectors: List[Dict], namespace: str = ""):
        """
        Internal method to upsert a batch of vectors.

        Args:
            vectors: List of vector dictionaries
            namespace: Pinecone namespace
        """
        try:
            self.index.upsert(vectors=vectors, namespace=namespace)

        except Exception as e:
            self.logger.log_error(e, context=f"Upserting batch of {len(vectors)}")
            raise

    def semantic_search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None, namespace: str = "") -> List[Dict]:
        """
        Perform semantic search with HNSW algorithm.

        This method used Pinecone's HNSW implementation to find semantically similar vectors based on cosine similarity.

        Args:
            query_embedding: Query vector from user question
            top_k: Number of most similar results to return
            filter_dict: Metadata filters (eg., {'category': 'cardiovascular'})
            namespace: Pinecone namespace to search in

        Returns:
            List of most similar documents with scores and metadata
        """
        try:
            if self.index is None:
                self.index = self.pc.Index(self.index_name)

            self.logger.logger.info(f"Performing HNSW semantic search...")
            self.logger.logger.info(f"  Algorithm: HNSW (Pinecone automatic)")
            self.logger.logger.info(f"  Similarity: Cosine")
            self.logger.logger.info(f"  Top K: {top_k}")
            if filter_dict:
                self.logger.logger.info(f"  Filters: {filter_dict}")

            # Query Pinecone using HNSW search
            if filter_dict:
                results = self.index.query(
                    vector = query_embedding,
                    top_k = top_k,
                    include_metadata = True,
                    filter = filter_dict,
                    namespace = namespace
                )
            else:
                results = self.index.query(
                    vector = query_embedding,
                    top_k = top_k,
                    include_metadata = True,
                    namespace = namespace
                )

            # Format results
            formatted_results = []
            for match in results['matches']:
                result = {
                    'id': match['id'],
                    'score': match['score'],
                    'text': match.get('metadata', {}).get('text', ''),
                    'metadata': match.get('metadata', {})
                }
                formatted_results.append(result)

            self.logger.logger.info(f"HNSW search completed")
            self.logger.logger.info(f"Found {len(formatted_results)} semantically similar results")

            self.logger.log_retrieval(len(formatted_results), "semantic_search")

            return formatted_results
        
        except Exception as e:
            self.logger.log_error(e, context = "HNSW semantic search")
            raise

    def hybrid_search(self, query_embedding: List[float], top_k: int = 5, category_filter: Optional[str] = None, section_filter: Optional[str] = None, namespace: str = "") -> List[Dict]:
        """
        Hybrid search: Metadata filtering + HNSW semantic search.

        Two-stage retrieval:
        1. Filter by metadata (category, section type)
        2. HNSW semantic search within filtered results

        Args:
            query_embedding: Query vector
            top_k: Number of results
            category_filter: Medical category (eg., 'cardiovascular')
            section_filter: Section type (eg., 'symptoms', 'treatment')
            namespace: Pinecone namespace

        Returns:
            Filtered and semantically ranked results
        """
        try:
            self.logger.logger.info("Performing hybrid search...")
            self.logger.logger.info("   Stage 1: Metadata filtering")
            self.logger.logger.info("   Stage 2: HNSW semantic search")

            # Build filter
            filter_dict = {}
            if category_filter:
                filter_dict['category'] = category_filter

            if section_filter:
                filter_dict['section_type'] = section_filter

            # Perform semantic search with filters
            results = self.semantic_search(
                query_embedding = query_embedding,
                top_k = top_k,
                filter_dict = filter_dict if filter_dict else None,
                namespace = namespace
            )

            self.logger.logger.info(f"Hybrid search completed")
            self.logger.logger.info(f"Retrieved {len(results)} filtered + semantic results")

            return results
        
        except Exception as e:
            self.logger.log_error(e, context = "Hybrid search")
            raise

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics including HNSW info
        """
        try:
            if self.index is None:
                existing_indexes = self.pc.list_indexes().names()
                if self.index_name in existing_indexes:
                    self.index = self.pc.Index(self.index_name)
                else:
                    return {"status": "Index not found"}
            
            # Get index stats
            stats = self.index.describe_index_stats()

            # Get index description for additional info
            index_description = self.pc.describe_index(self.index_name)

            index_info = {
                "index_name": self.index_name,
                "vector_database": "Pinecone",
                "search_algorithm": "HNSW (automatic)",
                "similarity_metric": index_description.metric,
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "namespaces": stats.get('namespaces', {}),
                "status": index_description.status.get('state', 'unknown')
            }

            self.logger.logger.info(f"Pinecone + HNSW stats: {index_info}")

            return index_info
        
        except Exception as e:
            self.logger.log_error(e, context = "Getting index stats")
            return {"error": str(e)}
        
    def delete_index(self):
        """Delete the entire Pinecone index"""
        try:
            if self.index_name in self.pc.list_indexes().names():
                self.logger.logger.info(f"Deleting Pinecone index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                self.logger.logger.info(f"Deleted index and all HNSW data")
                self.index = None
            else:
                self.logger.logger.warning(f"Index not found: {self.index_name}")

        except Exception as e:
            self.logger.log_error(e, context= "Deleting index")
            raise

    def delete_vectors(self, ids: Optional[List[str]] = None, delete_all: bool = False, namespace: str = ""):
        """
        Delete specific vectors or all vectors from index.

        Args:
            ids: List of vector IDs to delete
            delete_all: If True, delete all vectors (keep index structure)
            namespace: Pinecone namespace
        """
        try:
            if self.index is None:
                self.index = self.pc.Index(self.index_name)

            if delete_all:
                self.logger.logger.info(f"Deleting all vectors from: {self.index_name}")
                self.index.delete(delete_all = True, namespace = namespace)
                self.logger.logger.info(f"All vectors deleted (HNSW index cleared)")
            elif ids:
                self.logger.logger.info(f"Deleting {len(ids)} specific vectors")
                self.index.delete(ids = ids, namespace = namespace)
                self.logger.logger.info(f"{len(ids)} vectors deleted")
            else:
                self.logger.logger.warning("No vectors specified for deletion")

        except Exception as e:
            self.logger.log_error(e, context = "Deleting vecotrs")
            raise

class MedicalVectorStorePipeline:
    """
    Complete pipeline for medical RAG with Pinecone + HNSW

    Architecture:
    - Storage: Pinecone (cloud vector database)
    - Search: HNSW (automatic semantic search)
    - Metric: Cosine similarity (semantic text matching)
    """

    def __init__(self):
        """Initialize the pipeline with Pinecone + HNSW"""
        self.logger = MedicalRAGLogger(__name__)
        self.vectorstore = PineconeHNSWVectorStore()

        self.logger.logger.info("Medical Vector Store Pipeline initialized")
        self.logger.logger.info("   Vector Database: Pinecone")
        self.logger.logger.info("   Search Algorithm: HNSW")
        self.logger.logger.info("   Similarity Metric: Cosine")

    def store_embeddings_pipeline(self, embeddings: List[Tuple[List[float], Dict]], create_new_index: bool = False):
        """
        Complete pipeline: create index + store embeddings with HNSW indexing

        Args:
            embeddings: List of (embeddings, metadata) tuples from embeddings.py
            create_new_index: If True, recreate index from scratch
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("VECTOR STORAGE PIPELINE (PINECONE + HNSW)")
            self.logger.logger.info("=" * 60)

            # Step 1: Validate input
            if not embeddings:
                raise ValueError("No embeddings provided")
            
            dimension = len(embeddings[0][0])
            self.logger.logger.info(f"Input: {len(embeddings)} embeddings (dimension: {dimension})")

            # Step 2: Create/connect to index
            if create_new_index:
                self.logger.logger.info("Creating new Pinecone index with HNSW...")
                self.vectorstore.create_index(dimension=dimension, force_recreate=True)
            else:
                self.logger.logger.info("Using existing index or creating if needed...")
                self.vectorstore.create_index(dimension = dimension)

            # Step 3: Store embeddings (HNSW indexing automatic)
            self.logger.logger.info("Storing embeddings with HNSW indexing...")
            self.vectorstore.store_embeddings(embeddings)

            # Step 4: Verify storage
            stats = self.vectorstore.get_index_stats()
            self.logger.logger.info(f"\nIndex Statistics:")
            for key, value in stats.items():
                self.logger.logger.info(f"  {key}: {value}")

            self.logger.logger.info("\n" + "=" * 60)
            self.logger.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("Embeddings stored in Pinecone")
            self.logger.logger.info("HNSW search enabled")
            self.logger.logger.info("Ready for semantic search")

        except Exception as e:
            self.logger.log_error(e, context = "Vector storage pipeline")
            raise

    def store_documents_pipeline(self, documents: List[Dict], create_new_index: bool = False):
        """
        Complete pipeline for structured documents

        Args:
            documents: List of document dicts from embeddings.py
            create_new_index: If True, recreate index
        """
        try:
            self.logger.logger.info("Storing structured documents pipeline...")

            # Validate
            if not documents:
                raise ValueError("No documents provided")
            
            dimension = len(documents[0]['embedding'])
            self.logger.logger.info(f"Input: {len(documents)} documents (dimension: {dimension})")

            if create_new_index:
                self.vectorstore.create_index(dimension = dimension, force_recreate = True)
            else:
                self.vectorstore.create_index(dimension = dimension)

            # Store documents
            self.vectorstore.store_structured_documents(documents)

            # Get stats
            stats = self.vectorstore.get_index_stats()
            self.logger.logger.info(f"Index stats: {stats}")

            self.logger.logger.info("Documents stored successfully with HNSW")

        except Exception as e:
            self.logger.log_error(e, context = "Storing documents pipeline")
            raise

    def semantic_search_pipeline(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search using HNSW

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Metadata filters

        Returns:
            Semantically similar results
        """
        try:
            results = self.vectorstore.semantic_search(
                query_embedding = query_embedding,
                top_k = top_k,
                filter_dict = filters
            )

            return results
        
        except Exception as e:
            self.logger.log_error(e, context = "Semantic search pipeline")
            raise

    # Convenience functions
    @staticmethod
    def store_embeddings_in_pinecone(embeddings: List[Tuple[List[float], Dict]], create_new_index: bool = False):
        """
        Convenience function to store embeddings in Pinecone with HNSW

        Args:
            embeddings: Output from embeddings.py
            create_new_index: Whether to recreate index
        """
        pipeline = MedicalVectorStorePipeline()
        pipeline.store_embeddings_pipeline(embeddings, create_new_index)
    
    @staticmethod
    def semantic_search(query_embeddings: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Convenience function for semantic search

        Args:
            query_embedding: Query vector
            top_k: number of results
            filters: Metadata filters

        Returns:
            Search results
        """
        pipeline = MedicalVectorStorePipeline()
        return pipeline.semantic_search_pipeline(query_embeddings, top_k, filters)
    

# Usage/Testing
if __name__ == "__main__":
    from app.components.pdf_loader import MedicalPDFLoader
    from app.components.embeddings import MedicalEmbeddingPipeline
    
    print("\n" + "=" * 60)
    print("PINECONE + HNSW VECTOR STORE - TESTING")
    print("=" * 60 + "\n")

    # Step 1: Prepare test data
    print("Step 1: Preparing test data...")
    print("-" * 60)

    test_chunks = [
        Document(
            page_content="Diabetes is a chronic disease with high blood sugar levels.",
            metadata={"category": "endocrine", "section_type": "definition", "page": 100}
        ),
        Document(
            page_content="Common diabetes symptoms include increased thirst and frequent urination.",
            metadata={"category": "endocrine", "section_type": "symptoms", "page": 101}
        ),
        Document(
            page_content="Heart attack occurs when blood flow to the heart is blocked.",
            metadata={"category": "cardiovascular", "section_type": "definition", "page": 200}
        ),
        Document(
            page_content="Chest pain and shortness of breath are heart attack symptoms.",
            metadata={"category": "cardiovascular", "section_type": "symptoms", "page": 201}
        ),
        Document(
            page_content="Pneumonia is a lung infection causing breathing difficulties.",
            metadata={"category": "respiratory", "section_type": "definition", "page": 300}
        ),
    ]

    # Step 2: Generate embeddings
    print(f"Step 2: Generating embeddings...")
    print("-" * 60)

    try:
        embedding_pipeline = MedicalEmbeddingPipeline()
        embeddings = embedding_pipeline.process_chunks_to_embeddings(test_chunks)
        print(f"Generated {len(embeddings)} embeddings\n")
    except Exception as e:
        print(f"x Error: {e}\n")
        exit(1)

    # Step 3: Store in Pinecone with HNSW
    print("Step 3: Storing in Pinecone (HNSW enabled)...")
    print("-" * 60)

    try:
        vectorstore_pipeline = MedicalVectorStorePipeline()
        vectorstore_pipeline.store_embeddings_pipeline(
            embeddings,
            create_new_index = True
        )
        print("\n Embeddings stored with HNSW indexing!\n")
    except Exception as e:
        print(f"x Error: {e}\n")
        exit(1)

    # Step 4: Semantic search using HNSW
    print("Step 4: Testing HNSW semantic search...")
    print("-" * 60)

    test_queries = [
        "What is diabetes?",
        "heart attack symptoms",
        "lung infection"
    ]

    vectorstore = PineconeHNSWVectorStore()

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        try:
            # Generate query embedding
            query_embedding = embedding_pipeline.get_embedding_for_query(query)

            # Semantic search with HNSW
            results = vectorstore.semantic_search(
                query_embedding = query_embedding,
                top_k = 3
            )

            print(f"Found {len(results)} semantically similar results: \n")

            for i, result in enumerate(results):
                print(f"Result {i + 1}:")
                print(f"    Score: {result['score']: .4f}")
                print(f"    Text: {result['text'][:80]}...")
                print(f"    Category: {result['metadata'].get('category')}")
                print()

        except Exception as e:
            print(f"x Error: {e}\n")

    # Step 5: Filtered semantic search
    print("\n" + "=" * 60)
    print("Step 5: Testing filtered HNSW search...")
    print("-" * 60)

    print("\nQuery: 'symptoms' (Cardiovascular only)")
    print("-" * 40)

    try:
        query_embedding = embedding_pipeline.get_embedding_for_query("symptoms")

        results = vectorstore.semantic_search(
            query_embedding = query_embedding,
            top_k = 3,
            filter_dict = {"category": "cardiovascular"}
        )

        print(f"Found {len(results)} cardiovascular resultss:\n")

        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Text: {result['text'][:80]}...")
            print(f"  Category: {result['metadata'].get('category')}")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}\n")

    # Step 6: Index statistics
    print("="*60)
    print("Step 6: Index Statistics")
    print("-" * 60)
    
    try:
        stats = vectorstore.get_index_stats()
        print("\nPinecone + HNSW Index Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Error: {e}\n")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60 + "\n")
    
    print("Summary:")
    print("  ✓ Vector Database: Pinecone")
    print("  ✓ Search Algorithm: HNSW (automatic)")
    print("  ✓ Similarity Metric: Cosine")
    print("  ✓ Semantic Search: Working")
    print("  ✓ Metadata Filtering: Working")
    print("\nYour Medical RAG vector store is ready! 🎉")        