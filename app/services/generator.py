from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, model_name: str, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        
        logger.info(f"Loading generator model: {model_name}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info(f"Generator model loaded on {self.device}")
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on question and retrieved context"""
        
        # Construct prompt for T5
        prompt = f"""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=False
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    
    def estimate_confidence(self, answer: str, context: str) -> float:
        """Estimate confidence score based on answer and context overlap"""
        # Simple heuristic: measure word overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if len(answer_words) == 0:
            return 0.0
        
        overlap = len(answer_words.intersection(context_words))
        confidence = min(overlap / len(answer_words), 1.0)
        
        # Boost confidence if answer is reasonably long
        if len(answer.split()) > 5:
            confidence = min(confidence + 0.1, 1.0)
        
        return round(confidence, 2)