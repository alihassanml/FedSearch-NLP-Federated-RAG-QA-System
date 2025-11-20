from fastapi import APIRouter, HTTPException, status
from app.api.models import (
    QueryRequest, QueryResponse, 
    IndexRequest, IndexResponse,
    HealthResponse, RetrievedDocument
)
from app.core.rag_engine import RAGEngine
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global RAG engine instance
rag_engine: RAGEngine = None

def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine instance"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    engine = get_rag_engine()
    status_info = engine.get_status()
    
    return HealthResponse(
        status="healthy" if engine.is_ready() else "initializing",
        version=settings.APP_VERSION,
        models_loaded=status_info['initialized'],
        documents_indexed=status_info['retriever_stats'].get('total_documents', 0)
    )

@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Index or reindex company documents"""
    try:
        engine = get_rag_engine()
        result = engine.initialize(force_reindex=request.reindex)
        
        return IndexResponse(
            status=result['status'],
            documents_indexed=result['documents_indexed'],
            message=result['message']
        )
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index documents: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question"""
    try:
        engine = get_rag_engine()
        
        # Check if system is ready
        if not engine.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized. Please index documents first using /api/index endpoint."
            )
        
        # Process query
        result = engine.query(request.question, top_k=request.top_k)
        
        # Convert to response model
        retrieved_docs = [
            RetrievedDocument(
                content=doc['content'],
                score=doc['score'],
                source=doc['source']
            )
            for doc in result['retrieved_documents']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            retrieved_documents=retrieved_docs,
            confidence=result['confidence']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@router.get("/chat")
async def chat_interface():
    """Simple endpoint that can be used for testing"""
    return {
        "message": "Chat interface ready",
        "instructions": "Use POST /api/query with a JSON body containing 'question' field"
    }

@router.get("/documents/stats")
async def get_document_stats():
    """Get statistics about indexed documents"""
    try:
        engine = get_rag_engine()
        stats = engine.get_status()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )