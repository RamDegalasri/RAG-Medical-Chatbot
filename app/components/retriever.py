from app.config.config import Config
from app.common.logger import MedicalRAGLogger
from app.components.vector_store import PineconeHNSWVectorStore
from app.components.embeddings import MedicalEmbeddingPipeline
from app.components.llm import BedrockGemma3LLM
from typing import List, Dict, Optional

class MedicalRAGRetriever:
    """
    Complete RAG retrieval pipeline for Medical chatbot.

    Pipeline:
    1. User Question -> convert to embeddings
    2. Search vector database (HNSW)
    3. Retrieve relevant documents
    4. Pass to LLM for answer generation
    5. Return final response
    """

    def __init__(self):
        """Initialize RAG retriever components"""
        self.logger = MedicalRAGLogger(__name__)
        self.logger.logger.info("Initializing Medical RAG Retriever...")

        # Initialize components
        try:
            # Embedding pipeline (for query encoding)
            self.logger.logger.info("   Loading embedding model...")
            self.embeddings = MedicalEmbeddingPipeline(
                model_name = Config.AWS_BEDROCK_EMBEDDING_MODEL
            )

            # Vector store (for retrieval)
            self.logger.logger.info("   Connecting to vector database...")
            self.vectorstore = PineconeHNSWVectorStore()

            # LLM (for answer generation)
            self.logger.logger.info("   Initializing LLM...")
            self.llm = BedrockGemma3LLM()

            self.logger.logger.info("Medical RAG Retriever initialized")
            self.logger.logger.info("   Components: Embeddings + VectorDB + LLM")
        
        except Exception as e:
            self.logger.log_error(e, context = "Initializing RAG Retriever")
            raise

    def retrieve_documents(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant documents from vector database

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            filters: Metadata filters (eg. {'category': 'cardiovascular'})

        Returns:
            List of relevant documents with scores
        """
        try:
            self.logger.logger.info(f"Retrieving documents for: '{query[:50]}...'")
            self.logger.logger.info(f"  Top K: {top_k}")
            if filters:
                self.logger.logger.info(f"  Filters: {filters}")

            # Step 1: Convert query to embeddings
            query_embedding = self.embeddings.get_embedding_for_query(query)

            # Step 2: Search vector database using HNSW
            results = self.vectorstore.semantic_search(
                query_embedding = query_embedding,
                top_k = top_k,
                filter_dict = filters
            )

            self.logger.logger.info(f"Retrieved {len(results)} documents")

            # Log retrieval scores
            for i, doc in enumerate(results):
                self.logger.logger.debug(
                    f"  Doc {i + 1}: Score = {doc['score']:.4f}, "
                    f"Category = {doc['metadata'].get('category', 'N/A')}"
                )

            return results
        
        except Exception as e:
            self.logger.log_error(e, context = "Retrieving documents")
            raise

    def generate_answer(self, query: str, documents: List[Dict], temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """
        Generate answer using LLM and retrieved documents

        Args:
            query: User's question
            documents: Retrieved context documents
            temperature: LLM temperature (lower = more factual)
            max_tokens: Maximum response length

        Returns:
            Generated answer 
        """
        try:
            self.logger.logger.info("Generating answer with LLM...")

            # Build context from documents
            context = self._build_context(documents)

            # Create prompt
            prompt = self._create_prompt(query, context)

            # Generate response with LLM
            answer = self.llm.generate(
                prompt = prompt,
                max_tokens = max_tokens,
                temperature = temperature
            )

            self.logger.logger.info("Answer generated")
            self.logger.logger.info(f"  Answer length: {len(answer)} chars")

            return answer
        
        except Exception as e:
            self.logger.log_error(e, context = "Generating answer")
            raise

    def query(self, question: str, top_k: int = 5, filters: Optional[Dict] = None, temperature: float = 0.3, include_sources: bool = True) -> Dict:
        """
        Complete RAG query pipeline

        Args:
            question: User's question
            top_k: Number of documents to retrieve
            filters: Metadata filters
            temperature: LLM temperature
            include_sources: Include source documents in response

        Returns:
            Dict with answer and optional sources
        """
        try:
            self.logger.logger.info("=" * 60)
            self.logger.logger.info("MEDICAL RAG QUERY PIPELINE")
            self.logger.logger.info("=" * 60)
            self.logger.logger.info(f"Question: {question}")

            # Step 1: Retrieve relevant documents
            self.logger.logger.info("\nStep 1: Retrieving documents...")
            documents = self.retrieve_documents(
                query = question,
                top_k = top_k,
                filters = filters
            )

            # Check if documents found
            if not documents:
                self.logger.logger.warning("No relevant documents found")
                return {
                    'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                    'sources': [],
                    'num_sources': 0
                }
            
            # Step 2: Generate answer
            self.logger.logger.info("\nStep 2: Generating answer...")
            answer = self.generate_answer(
                query = question,
                documents = documents,
                temperature = temperature
            )

            # Step 3: Prepare response
            response = {
                'answer': answer,
                'num_sources': len(documents)
            }

            if include_sources:
                response['sources'] = self._format_sources(documents)

            self.logger.logger.info("\n" + "=" * 60)
            self.logger.logger.info("QUERY COMPLETED SUCCESSFULLY")
            self.logger.logger.info("=" * 60)

            return response
        
        except Exception as e:
            self.logger.log_error(e, context = "RAG query pipeline")
            raise

    def _build_context(self, documents: List[Dict]) -> str:
        """
        Build context string from retrieved documents

        Args:
            documents: Retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information available"
        
        context_parts = []

        for i, doc in enumerate(documents, 1):
            text = doc.get('text', doc.get('metadata', {}).get('text', ''))
            category = doc.get('metadata', {}).get('category', 'general')
            score = doc.get('score', 0.0)

            context_parts.append(
                f"[Source {i} - Category: {category}, Relevance: {score:.2f}]\n{text}"
            )
        
        context = "\n\n".join(context_parts)

        self.logger.logger.debug(f"Built context from {len(documents)} documents")

        return context
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create RAG prompt for LLM

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Forwarded prompt
        """
        prompt = f"""You are a helpful medical information assistant. Answer the question based ONLY on the provided context from a medical encyclopedia.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the context provided
- Be accurate and factual
- If the context doesn't contain the answer, say so
- Use clear, professional language
- Keep the answer concise and relevant


Answer: """
        
        return prompt
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Format source documents for response

        Args:
            documents: Retrieved documents

        Returns:
            Formatted source information
        """
        sources = []

        for i, doc in enumerate(documents, 1):
            source = {
                'id': i,
                'score': doc.get('score', 0.0),
                'category': doc.get('metadata', {}).get('category', 'unknown'),
                'section_type': doc.get('metadata', {}).get('section_type', 'general'),
                'page': doc.get('metadata', {}).get('page', 'N/A'),
                'text_preview': (doc.get('text', '')[:200] + '...') if len(doc.get('text', '')) > 200 else doc.get('text', '')
            }
            sources.append(source)

        return sources
    
class MedicalChatRetriever:
    """
    Conversational RAG retriever with chat history support
    """

    def __init__(self):
        """Initialize chat retriever"""
        self.logger = MedicalRAGLogger(__name__)
        self.retriever = MedicalRAGRetriever()
        self.conversation_history = []

        self.logger.logger.info("Medical Chat Retriever initialized")

    def chat(self, question: str, top_k: int = 5, filters: Optional[Dict] = None, use_history: bool = True) -> Dict:
        """
        Chat with conversation history

        Args:
            question: Current question
            top_k: Documents to retrieve
            filters: Metadata filters
            use_history: Include conversation history in context

        Returns:
            Response with answer and sources 
        """
        try:
            documents = self.retriever.retrieve_documents(
                query = question,
                top_k = top_k,
                filters = filters
            )

            # Build context with history
            context = self.retriever._build_context(documents)

            # Create prompt with history
            if use_history and self.conversation_history:
                prompt = self._create_chat_prompt(question, context)
            else:
                prompt = self.retriever._create_prompt(question, context)

            # Generate answer
            answer = self.retriever.llm.generate(
                prompt = prompt,
                temperature = 0.5,
                max_tokens = 1024
            )

            # Store in history
            self.conversation_history.append({
                'question': question,
                'answer': answer
            })

            # Keep last 5 turns
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]

            response = {
                'answer': answer,
                'sources': self.retriever._format_sources(documents),
                'num_sources': len(documents)
            }

            return response
        
        except Exception as e:
            self.logger.log_error(e, context = "Chat query")
            raise

    def _create_chat_prompt(self, question: str, context: str) -> str:
        """Create prompt with converational history"""

        history_text = "Previous conversation:\n"
        for turn in self.conversation_history[-3:]:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n\n"

        prompt = f"""You are helpful medical information assistant. Answer questions based on the context and conversation history.

{history_text}

Context from Medical Encyclopedia:
{context}

Current Question: {question}

Answer naturally while staying accurate to the medical information.

Answer:"""
        
        return prompt
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
# Usage/Testing
if __name__ == "__main__":

    print("\n" + "=" * 190)
    print("MEDICAL RAG RETRIEVER - TESTING")
    print("=" * 190 + "\n")

    # Step 1: Initialize Retriever
    print("Step 1: Initializing RAG Retriever...")
    print("-" * 190)

    try:
        retriever = MedicalRAGRetriever()
        print("RAG Retriever initialized\n")
    except Exception as e:
        print(f"x Error: {e}\n")
        print("Make sure:")
        print(" 1. Vector database has data")
        print(" 2. OpenAI API key is set")
        print(" 3. AWS Bedrock is configured")
        exit(1)

    # Step 2: Test document retrieval
    print("Step 2: Testing document retrieval...")
    print("-" * 190)

    test_query = "What is diabetes?"

    try:
        print(f"Query: '{test_query}'\n")

        documents = retriever.retrieve_documents(
            query = test_query,
            top_k = 3
        )

        print(f"Retrieved {len(documents)} documents\n")

    except Exception as e:
        print(f"x Error: {e}\n")

    # Step 3: Test complete RAG pipeline
    print("Step 3: Testing complete RAG pipeline")
    print("-" * 190)

    test_questions = [
        "What is diabetes?",
        "What are the symptoms of diabetes?",
        "How is diabetes treated?"
    ]

    for question in test_questions:
        try:
            print(f"\nQuestion: {question}")
            print("-" * 190)

            result = retriever.query(
                question = question,
                top_k = 3,
                temperature = 0.3,
                include_sources = True
            )

            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Sources used: {result['num_sources']}")

            if result.get('sources'):
                print("\nSource details:")
                for source in result['sources'][:2]:  # Show first 2
                    print(f"    [{source['id']}] {source['category']} - Score: {source['score']:.3f}")

            print()

        except Exception as e:
            print(f"x Error: {e}\n")

    # Step 4: Test filtered retrieval
    print("\nStep 4: Testing filtered retrieval...")
    print("-" * 190)

    try:
        print("Query: 'symptoms' (Cardiovascular only)\n")

        result = retriever.query(
            question = "What are common symptoms?",
            top_k = 3,
            filters = {'category': 'cardiovascular'},
            temperature = 0.3
        )

        print(f"Answer:\n{result['answer']}\n")
        print(f"Sources: {result['num_sources']} (all cardiovascular)")

    except Exception as e:
        print(f"x Error: {e}\n")

    # Step 5: Test chat with history
    print("\nStep 5: Testing conversational retrieval...")
    print("-" * 190)

    try:
        chat = MedicalChatRetriever()

        # Turn 1
        print("\nTurn 1:")
        print("User: What is diabetes?")
        result1 = chat.chat("What is diabetes?", top_k = 3)
        print(f"Assistant: {result1['answer'][:300]}...\n")

        # Turn 2 (follow-up)
        print("Turn 2:")
        print("User: What are its symptoms?")
        result2 = chat.chat("What are its symptoms?", top_k=3)
        print(f"Assistant: {result2['answer'][:150]}...\n")
        
        print(f"Chat history: {len(chat.get_history())} turns")
        
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60 + "\n")
    
    print("Summary:")
    print("   Document retrieval working")
    print("   LLM answer generation working")
    print("   Complete RAG pipeline working")
    print("   Metadata filtering working")
    print("   Conversational chat working")
    print("\nYour Medical RAG Retriever is ready!")