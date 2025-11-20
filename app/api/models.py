from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question to the system")
    top_k: Optional[int] = Field(3, description="Number of documents to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the annual leave policy?",
                "top_k": 3
            }
        }

class RetrievedDocument(BaseModel):
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    source: str = Field(..., description="Document source/filename")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    retrieved_documents: List[RetrievedDocument] = Field(..., description="Retrieved context documents")
    confidence: float = Field(..., description="Confidence score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Annual leave is 20 days per year for full-time employees.",
                "retrieved_documents": [
                    {
                        "content": "Annual Leave: 20 days per year...",
                        "score": 0.89,
                        "source": "hr_policy.txt"
                    }
                ],
                "confidence": 0.92
            }
        }

class IndexRequest(BaseModel):
    reindex: bool = Field(False, description="Force reindexing of all documents")

class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    documents_indexed: int