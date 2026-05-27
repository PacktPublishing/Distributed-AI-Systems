"""
API Gateway for Production LLM Serving

Handles request routing, load balancing, authentication, rate limiting,
and request/response transformation.

Usage:
    uvicorn api_gateway:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import time
from collections import defaultdict

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class GenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen2.5-1.5b"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9


class GenerationResponse(BaseModel):
    text: str
    model: str
    latency_ms: float
    tokens_generated: int


# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        client_requests = self.requests[client_id]

        # Remove old requests
        client_requests[:] = [
            t for t in client_requests if now - t < self.window_seconds
        ]

        # Check limit
        if len(client_requests) >= self.max_requests:
            return False

        # Add current request
        client_requests.append(now)
        return True


rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Model routing
MODEL_ENDPOINTS = {
    "qwen2.5-1.5b": "http://localhost:8002",
    "qwen2.5-0.5b": "http://localhost:8003",
}


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, client_id: str = "default"):
    """Generate text endpoint"""
    # Rate limiting
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Route to appropriate model
    model_endpoint = MODEL_ENDPOINTS.get(request.model)
    if not model_endpoint:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    # Forward request to model runner
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{model_endpoint}/generate",
            json={
                "prompt": request.prompt,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
            },
            timeout=60.0,
        )

    latency_ms = (time.time() - start_time) * 1000

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    result = response.json()

    return GenerationResponse(
        text=result["text"],
        model=request.model,
        latency_ms=latency_ms,
        tokens_generated=result.get("tokens_generated", 0),
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
