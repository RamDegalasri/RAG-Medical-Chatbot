# app/application.py

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app.common.logger import MedicalRAGLogger
from app.components.retriever import MedicalRAGRetriever, MedicalChatRetriever
import traceback


# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.jinja_env.variable_start_string = '[['
app.jinja_env.variable_end_string = ']]'
CORS(app)  # Enable CORS for API calls

# Initialize logger
logger = MedicalRAGLogger(__name__)

# Initialize RAG components
logger.logger.info("Initializing Medical RAG application...")

try:
    rag_retriever = MedicalRAGRetriever()
    chat_sessions = {}  # Store chat sessions by session_id
    logger.logger.info("✓ Medical RAG application initialized")
except Exception as e:
    logger.log_error(e, context="Initializing application")
    raise


# Routes
@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Medical RAG Chatbot',
        'version': '1.0.0'
    })


@app.route('/api/query', methods=['POST'])
def query():
    """
    Query endpoint for RAG
    
    Request JSON:
    {
        "question": "What is diabetes?",
        "top_k": 5,
        "filters": {"category": "endocrine"},
        "temperature": 0.3,
        "include_sources": true
    }
    
    Response JSON:
    {
        "success": true,
        "answer": "...",
        "sources": [...],
        "num_sources": 5
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400
        
        question = data['question']
        top_k = data.get('top_k', 5)
        filters = data.get('filters', None)
        temperature = data.get('temperature', 0.3)
        include_sources = data.get('include_sources', True)
        
        logger.logger.info(f"Query received: {question}")
        
        # Query RAG system
        result = rag_retriever.query(
            question=question,
            top_k=top_k,
            filters=filters,
            temperature=temperature,
            include_sources=include_sources
        )
        
        # Return response
        response = {
            'success': True,
            'answer': result['answer'],
            'num_sources': result['num_sources']
        }
        
        if include_sources:
            response['sources'] = result.get('sources', [])
        
        logger.logger.info("Query completed successfully")
        
        return jsonify(response)
        
    except Exception as e:
        logger.log_error(e, context="Query endpoint")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint with conversation history
    
    Request JSON:
    {
        "session_id": "unique-session-id",
        "question": "What is diabetes?",
        "top_k": 5,
        "filters": null
    }
    
    Response JSON:
    {
        "success": true,
        "answer": "...",
        "sources": [...],
        "session_id": "unique-session-id"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400
        
        question = data['question']
        session_id = data.get('session_id', 'default')
        top_k = data.get('top_k', 5)
        filters = data.get('filters', None)
        
        logger.logger.info(f"Chat query received: {question} (session: {session_id})")
        
        # Get or create chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = MedicalChatRetriever()
        
        chat_retriever = chat_sessions[session_id]
        
        # Query with history
        result = chat_retriever.chat(
            question=question,
            top_k=top_k,
            filters=filters,
            use_history=True
        )
        
        # Return response
        response = {
            'success': True,
            'answer': result['answer'],
            'sources': result.get('sources', []),
            'num_sources': result['num_sources'],
            'session_id': session_id
        }
        
        logger.logger.info("Chat query completed successfully")
        
        return jsonify(response)
        
    except Exception as e:
        logger.log_error(e, context="Chat endpoint")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """
    Clear chat history
    
    Request JSON:
    {
        "session_id": "unique-session-id"
    }
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_sessions:
            chat_sessions[session_id].clear_history()
            logger.logger.info(f"Chat history cleared for session: {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        })
        
    except Exception as e:
        logger.log_error(e, context="Clear chat endpoint")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/retrieve', methods=['POST'])
def retrieve_documents():
    """
    Retrieve documents only (no LLM generation)
    
    Request JSON:
    {
        "query": "diabetes",
        "top_k": 5,
        "filters": {"category": "endocrine"}
    }
    
    Response JSON:
    {
        "success": true,
        "documents": [...],
        "num_documents": 5
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        filters = data.get('filters', None)
        
        logger.logger.info(f"Retrieve request: {query}")
        
        # Retrieve documents
        documents = rag_retriever.retrieve_documents(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        return jsonify({
            'success': True,
            'documents': documents,
            'num_documents': len(documents)
        })
        
    except Exception as e:
        logger.log_error(e, context="Retrieve endpoint")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available medical categories"""
    categories = [
        'cardiovascular',
        'neurological',
        'respiratory',
        'endocrine',
        'gastrointestinal',
        'musculoskeletal',
        'dermatological',
        'psychiatric',
        'immunological',
        'reproductive',
        'urological',
        'oncological'
    ]
    
    return jsonify({
        'success': True,
        'categories': categories
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.log_error(error, context="Internal server error")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# Run application
if __name__ == '__main__':
    logger.logger.info("Starting Medical RAG Flask application...")
    logger.logger.info("  Host: 0.0.0.0")
    logger.logger.info("  Port: 5001")
    logger.logger.info("  Debug: True")

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )