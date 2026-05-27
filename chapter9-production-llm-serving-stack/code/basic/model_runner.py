"""
Model Runner for Production LLM Serving

A stateful, GPU-backed inference engine that manages model loading, KV cache, 
and batching. Runs as a FastAPI server that the API gateway can route to.

Usage:
    uvicorn model_runner:app --host 0.0.0.0 --port 8002
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from vllm import LLM, SamplingParams

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


class ModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = None

    def load_model(self):
        """Load model for inference"""
        if self.llm is not None:
            return
        print(f"Loading model {self.model_name}...")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
        )
        print("Model loaded successfully")

    def generate(self, prompt: str, **kwargs) -> tuple[str, int]:
        """Generate text from prompt"""
        if self.llm is None:
            self.load_model()

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 512),
        )

        outputs = self.llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        return output_text, tokens_generated


# Initialize model runner (lazy loading)
runner = ModelRunner("Qwen/Qwen2.5-1.5B-Instruct")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    runner.load_model()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text endpoint"""
    try:
        text, tokens = runner.generate(
            request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        return GenerateResponse(text=text, tokens_generated=tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": runner.model_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
