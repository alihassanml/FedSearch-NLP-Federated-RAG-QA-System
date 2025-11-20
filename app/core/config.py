from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "FedSearch-NLP"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model Configuration
    RETRIEVER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATOR_MODEL: str = "google/flan-t5-base"
    VECTOR_DIM: int = 384
    TOP_K: int = 3
    
    # Data Paths
    COMPANY_DOCS_PATH: str = "data/company_docs"
    EMBEDDINGS_PATH: str = "data/embeddings"
    MODELS_CACHE_PATH: str = "models"
    
    # Performance
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.COMPANY_DOCS_PATH,
            self.EMBEDDINGS_PATH,
            self.MODELS_CACHE_PATH
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

settings = Settings()
settings.ensure_directories()