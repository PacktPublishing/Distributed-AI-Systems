# 09/code/k3d/manage-cluster-multi-models.sh

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Routes to appropriate vLLM service                       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service 1      │    │  vLLM Service 2     │
    │  (Llama-3.2-1B)      │    │  (Phi-tiny-MoE)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: vllm-phi-tiny │
    │ Service: vllm-llama- │    │  Service: vllm-phi- │
    │    32-1b:8000        │    │    tiny-moe:8000    │
    └──────────────────────┘    └─────────────────────┘
```


# 09/code/k3d/manage-cluster-multi-engines.sh

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' and 
                            │ 'owned_by' fields
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Parses 'owned_by' field                                  │
│  - Routes to appropriate service based on both fields       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service        │    │  SGLang Service     │
    │  (Llama-3.2-1B)      │    │  (Llama-3.2-1B)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: sglang-llama- │
    │ Service: vllm-llama- │    │    32-1b            │
    │    32-1b:8000        │    │  Service: sglang-   │
    │                      │    │    llama-32-1b:8000 │
    │ owned_by: "vllm"     │    │  owned_by: "sglang" │
    │                      │    │                     │
    │ Image: vllm/vllm-    │    │  Image: lmsysorg/   │
    │   openai:v0.12.0     │    │    sglang:v0.5.6    │
    └──────────────────────┘    └─────────────────────┘
```