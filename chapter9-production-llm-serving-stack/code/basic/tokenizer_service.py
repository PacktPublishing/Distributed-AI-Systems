"""
Tokenizer Service for Production AI Serving

A stateless, lightweight service that handles text tokenization and detokenization.
Can be scaled independently with low latency requirement (<10ms).

This service is useful for:
- Diffusion models that need CLIP tokenization before text encoding
- Counting tokens for billing or rate limiting before inference
- Custom inference backends that don't include tokenization

Usage:
    uvicorn tokenizer_service:app --host 0.0.0.0 --port 8001

Example requests:
    # LLM tokenization
    curl -X POST http://localhost:8001/tokenize \
      -H "Content-Type: application/json" \
      -d '{"model": "qwen2.5-1.5b", "text": "Hello world"}'

    # Diffusion model tokenization (CLIP)
    curl -X POST http://localhost:8001/tokenize \
      -H "Content-Type: application/json" \
      -d '{"model": "stable-diffusion", "text": "a photo of a cat"}'
"""

import os
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, CLIPTokenizer

app = FastAPI()

DEBUG = os.environ.get("DEBUG", "").upper() == "TRUE"


class TokenizerService:
    def __init__(self):
        self.tokenizers = {}
        self.load_llm_tokenizer("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct")
        self.load_clip_tokenizer("stable-diffusion", "openai/clip-vit-large-patch14")

    def load_llm_tokenizer(self, model_name, tokenizer_path):
        """Load tokenizer for LLM models"""
        if DEBUG:
            print(f"[DEBUG] Loading LLM tokenizer: {model_name} from {tokenizer_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizers[model_name] = tokenizer
        if DEBUG:
            print(f"[DEBUG] Tokenizer {model_name} loaded, vocab size: {tokenizer.vocab_size}", flush=True)

    def load_clip_tokenizer(self, model_name, tokenizer_path):
        """Load CLIP tokenizer for diffusion models"""
        if DEBUG:
            print(f"[DEBUG] Loading CLIP tokenizer: {model_name} from {tokenizer_path}", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        self.tokenizers[model_name] = tokenizer
        if DEBUG:
            print(f"[DEBUG] CLIP tokenizer {model_name} loaded, vocab size: {tokenizer.vocab_size}", flush=True)

    def encode(self, model_name, text):
        """Tokenize text"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found. "
                           f"Available: {list(self.tokenizers.keys())}")

        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.encode(text, return_tensors="pt")
        return tokens.tolist()[0]

    def decode(self, model_name, token_ids):
        """Detokenize tokens"""
        if model_name not in self.tokenizers:
            raise ValueError(f"Tokenizer for {model_name} not found. "
                           f"Available: {list(self.tokenizers.keys())}")

        tokenizer = self.tokenizers[model_name]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return text

    def count_tokens(self, model_name, text):
        """Count tokens without returning the full token list (for billing)"""
        tokens = self.encode(model_name, text)
        return len(tokens)


tokenizer_service = TokenizerService()


@app.post("/tokenize")
async def tokenize(request: dict):
    """Tokenize endpoint"""
    model_name = request.get("model", "qwen2.5-1.5b")
    text = request.get("text", "")

    if DEBUG:
        print(f"[DEBUG] /tokenize request: model={model_name}, text_len={len(text)}")

    try:
        tokens = tokenizer_service.encode(model_name, text)
        if DEBUG:
            print(f"[DEBUG] /tokenize response: {len(tokens)} tokens")
        return {"tokens": tokens, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detokenize")
async def detokenize(request: dict):
    """Detokenize endpoint"""
    model_name = request.get("model", "qwen2.5-1.5b")
    token_ids = request.get("tokens", [])

    if DEBUG:
        print(f"[DEBUG] /detokenize request: model={model_name}, num_tokens={len(token_ids)}", flush=True)

    try:
        text = tokenizer_service.decode(model_name, token_ids)
        if DEBUG:
            print(f"[DEBUG] /detokenize response: text_len={len(text)}", flush=True)
        return {"text": text, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/count")
async def count_tokens(request: dict):
    """Count tokens endpoint (useful for billing/rate limiting)"""
    model_name = request.get("model", "qwen2.5-1.5b")
    text = request.get("text", "")

    if DEBUG:
        print(f"[DEBUG] /count request: model={model_name}, text_len={len(text)}", flush=True)

    try:
        count = tokenizer_service.count_tokens(model_name, text)
        if DEBUG:
            print(f"[DEBUG] /count response: {count} tokens", flush=True)
        return {"token_count": count, "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available tokenizers"""
    return {"models": list(tokenizer_service.tokenizers.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
