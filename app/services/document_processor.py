import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        
    def load_documents(self) -> List[Dict[str, str]]:
        """Load all text documents from the company docs directory"""
        documents = []
        
        if not os.path.exists(self.docs_path):
            logger.warning(f"Documents path does not exist: {self.docs_path}")
            return documents
        
        for filename in os.listdir(self.docs_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.docs_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Split into chunks (simple paragraph-based splitting)
                    chunks = self._split_into_chunks(content)
                    
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            'id': f"{filename}_{i}",
                            'content': chunk,
                            'source': filename,
                            'chunk_index': i
                        })
                    
                    logger.info(f"Loaded {len(chunks)} chunks from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks based on paragraphs and size"""
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk_size, save current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_document_stats(self) -> Dict:
        """Get statistics about loaded documents"""
        documents = self.load_documents()
        
        sources = {}
        for doc in documents:
            source = doc['source']
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_chunks': len(documents),
            'unique_sources': len(sources),
            'sources': sources
        }