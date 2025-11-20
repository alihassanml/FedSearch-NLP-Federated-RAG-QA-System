import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import logging

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"


logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, model_name: str, vector_dim: int, embeddings_path: str):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.embeddings_path = embeddings_path
        self.index_file = os.path.join(embeddings_path, "faiss_index.bin")
        self.docs_file = os.path.join(embeddings_path, "documents.pkl")
        
        # Initialize model
        logger.info(f"Loading retriever model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """Build FAISS index from documents"""
        logger.info(f"Building index for {len(documents)} documents...")
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents = documents
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Index built successfully with {len(documents)} documents")
    
    def load_index(self) -> bool:
        """Load existing FAISS index from disk"""
        if not os.path.exists(self.index_file) or not os.path.exists(self.docs_file):
            logger.warning("Index files not found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_file)
            
            # Load documents
            with open(self.docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Index loaded successfully with {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def _save_index(self) -> None:
        """Save FAISS index to disk"""
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)
        
        # Save documents
        with open(self.docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info("Index saved to disk")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """Search for relevant documents"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("Index not initialized")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (inverse)
                similarity_score = 1 / (1 + dist)
                results.append((self.documents[idx], similarity_score))
        
        return results
    
    def is_indexed(self) -> bool:
        """Check if index is ready"""
        return self.index is not None and len(self.documents) > 0
    
    def get_stats(self) -> Dict:
        """Get retriever statistics"""
        return {
            'indexed': self.is_indexed(),
            'total_documents': len(self.documents),
            'model': self.model_name,
            'vector_dim': self.vector_dim
        }