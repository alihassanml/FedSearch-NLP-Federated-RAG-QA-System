"""
Federated Learning Client
Each department/company runs this
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, List, Optional
import logging
import copy

from app.services.retriever import DocumentRetriever
from app.services.generator import AnswerGenerator
from app.federated.dp_mechanism import DifferentialPrivacy

logger = logging.getLogger(__name__)

class FederatedClient:
    """
    Federated client for local training
    Each company/department runs one instance
    """
    
    def __init__(
        self,
        client_id: str,
        data_path: str,
        retriever_model: str,
        generator_model: str,
        use_dp: bool = True,
        epsilon: float = 1.0,
        learning_rate: float = 1e-4
    ):
        self.client_id = client_id
        self.data_path = data_path
        self.use_dp = use_dp
        
        logger.info(f"Initializing FL client: {client_id}")
        
        # Initialize models
        self.retriever = DocumentRetriever(
            model_name=retriever_model,
            vector_dim=384,
            embeddings_path=f"data/embeddings_{client_id}"
        )
        
        self.generator = AnswerGenerator(
            model_name=generator_model,
            max_length=512
        )
        
        # Differential Privacy
        if use_dp:
            self.dp = DifferentialPrivacy(epsilon=epsilon)
        else:
            self.dp = None
        
        # Optimizer
        self.optimizer = AdamW(
            self.generator.model.parameters(),
            lr=learning_rate
        )
        
        # Training stats
        self.local_epochs = 0
        self.data_size = 0
    
    def load_local_data(self):
        """Load and index local documents"""
        logger.info(f"Client {self.client_id}: Loading local data from {self.data_path}")
        
        from app.services.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(self.data_path)
        documents = processor.load_documents()
        
        self.data_size = len(documents)
        logger.info(f"Client {self.client_id}: Loaded {self.data_size} documents")
        
        # Build local index
        self.retriever.build_index(documents)
        
        return documents
    
    def local_train(
        self,
        epochs: int = 5,
        batch_size: int = 4,
        global_model: Optional[Dict] = None
    ) -> Dict:
        """
        Perform local training on private data
        
        Args:
            epochs: Number of local epochs
            batch_size: Training batch size
            global_model: Global model weights from server
        
        Returns:
            Updated local model weights
        """
        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs")
        
        # Apply global model if provided
        if global_model is not None:
            self.apply_global_model(global_model)
        
        # Training mode
        self.generator.model.train()
        
        # Create training samples (simplified)
        train_samples = self._create_training_samples()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(0, len(train_samples), batch_size):
                batch = train_samples[i:i + batch_size]
                
                # Forward pass
                loss = self._train_batch(batch)
                
                # Backward pass
                loss.backward()
                
                # Apply DP if enabled
                if self.dp:
                    # Clip gradients
                    self.dp.clip_gradients(
                        self.generator.model.parameters()
                    )
                    
                    # Add noise
                    self.dp.add_noise(
                        self.generator.model.parameters()
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (len(train_samples) / batch_size)
            losses.append(avg_loss)
            
            logger.info(f"Client {self.client_id}, Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        self.local_epochs += epochs
        
        # Track privacy if using DP
        if self.dp:
            privacy_spent = self.dp.track_privacy(epochs * len(train_samples))
            logger.info(f"Client {self.client_id}: Privacy spent = ε {privacy_spent:.4f}")
        
        return self.get_model_updates()
    
    def _create_training_samples(self) -> List[Dict]:
        """Create training samples from local documents"""
        # Simplified: Generate question-answer pairs
        samples = []
        
        # In practice, use the indexed documents to create QA pairs
        # For now, return dummy samples
        for i in range(10):  # Simulate 10 training samples
            samples.append({
                'question': f'Sample question {i} for {self.client_id}',
                'context': f'Sample context {i}',
                'answer': f'Sample answer {i}'
            })
        
        return samples
    
    def _train_batch(self, batch: List[Dict]) -> torch.Tensor:
        """Train on a single batch"""
        # Simplified training
        # In practice, properly encode inputs and compute loss
        
        total_loss = 0
        
        for sample in batch:
            # Create prompt
            prompt = f"question: {sample['question']} context: {sample['context']}"
            
            # Tokenize
            inputs = self.generator.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.generator.device)
            
            labels = self.generator.tokenizer(
                sample['answer'],
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).input_ids.to(self.generator.device)
            
            # Forward pass
            outputs = self.generator.model(**inputs, labels=labels)
            loss = outputs.loss
            
            total_loss += loss
        
        return total_loss / len(batch)
    
    def get_model_updates(self) -> Dict:
        """Get model parameters to send to server"""
        # Only send adapter/trainable parameters
        model_state = {}
        
        for name, param in self.generator.model.named_parameters():
            if param.requires_grad:
                model_state[name] = param.data.cpu().clone()
        
        logger.info(f"Client {self.client_id}: Prepared {len(model_state)} parameters for upload")
        
        return model_state
    
    def apply_global_model(self, global_model: Dict):
        """Apply global model from server"""
        logger.info(f"Client {self.client_id}: Applying global model")
        
        model_dict = self.generator.model.state_dict()
        
        # Update only the parameters that were trained
        for name, param in global_model.items():
            if name in model_dict:
                model_dict[name] = param.to(self.generator.device)
        
        self.generator.model.load_state_dict(model_dict, strict=False)
        
        logger.info(f"Client {self.client_id}: Global model applied")
    
    def evaluate(self, test_samples: List[Dict]) -> Dict:
        """Evaluate local model"""
        self.generator.model.eval()
        
        correct = 0
        total = len(test_samples)
        
        with torch.no_grad():
            for sample in test_samples:
                # Generate answer
                generated = self.generator.generate_answer(
                    sample['question'],
                    sample['context']
                )
                
                # Simple accuracy check (in practice, use better metrics)
                if sample['answer'].lower() in generated.lower():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Client {self.client_id}: Accuracy = {accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }