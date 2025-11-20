from typing import Dict, List
import logging
from app.services.document_processor import DocumentProcessor
from app.services.retriever import DocumentRetriever
from app.services.generator import AnswerGenerator
from app.core.config import settings
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        self.document_processor = DocumentProcessor(settings.COMPANY_DOCS_PATH)
        self.retriever = DocumentRetriever(
            model_name=settings.RETRIEVER_MODEL,
            vector_dim=settings.VECTOR_DIM,
            embeddings_path=settings.EMBEDDINGS_PATH
        )
        self.generator = AnswerGenerator(
            model_name=settings.GENERATOR_MODEL,
            max_length=settings.MAX_LENGTH
        )
        self._initialized = False
    
    def initialize(self, force_reindex: bool = False) -> Dict:
        """Initialize the RAG system by loading or building the index"""
        logger.info("Initializing RAG Engine...")
        
        # Try to load existing index
        if not force_reindex and self.retriever.load_index():
            logger.info("Loaded existing index")
            self._initialized = True
            return {
                'status': 'success',
                'message': 'Loaded existing index',
                'documents_indexed': len(self.retriever.documents)
            }
        
        # Build new index
        logger.info("Building new index...")
        documents = self.document_processor.load_documents()
        
        if len(documents) == 0:
            logger.error("No documents found to index")
            return {
                'status': 'error',
                'message': 'No documents found in data/company_docs',
                'documents_indexed': 0
            }
        
        self.retriever.build_index(documents)
        self._initialized = True
        
        return {
            'status': 'success',
            'message': 'Index built successfully',
            'documents_indexed': len(documents)
        }
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Process a query through the RAG pipeline"""
        if not self._initialized:
            logger.error("RAG Engine not initialized")
            return {
                'answer': 'System not initialized. Please index documents first.',
                'retrieved_documents': [],
                'confidence': 0.0
            }
        
        # Step 1: Retrieve relevant documents
        logger.info(f"Processing query: {question}")
        retrieved_docs = self.retriever.search(question, top_k=top_k)
        
        if len(retrieved_docs) == 0:
            return {
                'answer': 'No relevant information found in the knowledge base.',
                'retrieved_documents': [],
                'confidence': 0.0
            }
        
        # Step 2: Prepare context
        context = "\n\n".join([doc['content'] for doc, _ in retrieved_docs])
        
        # Step 3: Generate answer
        answer = self.generator.generate_answer(question, context)
        
        # Step 4: Estimate confidence
        confidence = self.generator.estimate_confidence(answer, context)
        
        # Prepare response
        retrieved_documents = [
            {
                'content': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content'],
                'score': float(score),
                'source': doc['source']
            }
            for doc, score in retrieved_docs
        ]
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_documents,
            'confidence': confidence
        }
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            'initialized': self._initialized,
            'retriever_stats': self.retriever.get_stats(),
            'document_stats': self.document_processor.get_document_stats()
        }
    
    def is_ready(self) -> bool:
        """Check if system is ready to accept queries"""
        return self._initialized and self.retriever.is_indexed()