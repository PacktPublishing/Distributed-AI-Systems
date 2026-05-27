"""
Tokenizer Service for LLM Serving Stack.
"""
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import asyncio

app = FastAPI()

class TokenizerService:
    def __init__(self):
        self.tokenizers = {}
        self.load_tokenizer("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct")
    
    def load_tokenizer(self, model_name, tokenizer_path):
        """Load tokenizer for a model"""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizers[model_name] = tokenizer
    
    def encode(self, model_name, text):
        """Tokenize text"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found")
        
        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.encode(text, return_tensors="pt")
        return tokens.tolist()[0]
    
    def decode(self, model_name, token_ids):
        """Detokenize tokens"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found")
        
        tokenizer = self.tokenizers[model_name]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

tokenizer_service = TokenizerService()

@app.post("/tokenize")
async def tokenize(request: dict):
    """Tokenize endpoint"""
    model_name = request.get("model", "qwen2.5-1.5b")
    text = request.get("text", "")
    
    try:
        tokens = tokenizer_service.encode(model_name, text)
        return {"tokens": tokens, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detokenize")
async def detokenize(request: dict):
    """Detokenize endpoint"""
    model_name = request.get("model", "qwen2.5-1.5b")
    token_ids = request.get("tokens", [])
    
    try:
        text = tokenizer_service.decode(model_name, token_ids)
        return {"text": text, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

