"""
llm-d API Gateway for Engine Comparison - Routes requests to LLMInferenceService based on 'model' and 'owned_by' fields

This gateway supports routing by both model name and inference engine type (owned_by).
It can read 'owned_by' from request body or HTTP header (x-owned-by).
Adapted from llm-d-multi-model/api-gateway.py for LLMInferenceService CRDs.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="llm-d Engine Comparison API Gateway")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service port (all LLMInferenceServices use 8000)
SERVICE_PORT = 8000
NAMESPACE = os.getenv("NAMESPACE", "llm-d-multi-engine")

# Initialize Kubernetes client
try:
    config.load_incluster_config()
    logger.info("Loaded in-cluster Kubernetes config")
except:
    try:
        config.load_kube_config()
        logger.info("Loaded kubeconfig from default location")
    except Exception as e:
        logger.warning(f"Could not load Kubernetes config: {e}. Service discovery will be limited.")

v1 = client.CoreV1Api()

# Routing configuration (can be updated dynamically)
# Format: {(model_name, owned_by): service_name}
# owned_by can be "vllm", "sglang", etc., or None for default
ROUTING_CONFIG: Dict[Tuple[str, Optional[str]], str] = {
    # Will be auto-populated from Kubernetes Services
}

# Cache pod IPs by model name for direct access
POD_IP_CACHE: Dict[str, str] = {}

async def discover_services():
    """Discover LLMInferenceService instances from Kubernetes"""
    try:
        import asyncio
        
        # Discover from pods with ModelService label (llm-d.ai/role=decode)
        # ModelService pods have label: llm-d.ai/role=decode
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="llm-d.ai/role=decode"
        )
        
        discovered = {}
        pod_to_service = {}
        
        # First, get all services to map pod IPs to service names
        # For ModelService, we need to find services that match the pods
        services = v1.list_namespaced_service(namespace=NAMESPACE)
        for svc in services.items:
            try:
                endpoints = v1.read_namespaced_endpoints(svc.metadata.name, NAMESPACE)
                for subset in endpoints.subsets or []:
                    for address in subset.addresses or []:
                        if address.target_ref and address.target_ref.kind == "Pod":
                            pod_to_service[address.ip] = svc.metadata.name
                            # Also map by pod name for better matching
                            if address.target_ref.name:
                                pod_to_service[address.target_ref.name] = svc.metadata.name
            except Exception as e:
                logger.warning(f"Failed to get endpoints for service {svc.metadata.name}: {e}")
        
        async def query_single_pod(pod, pod_ip, pod_to_service):
            """Query a single pod to get its model name"""
            # Try to find service by pod IP or pod name
            service_name = pod_to_service.get(pod_ip, "")
            if not service_name:
                service_name = pod_to_service.get(pod.metadata.name, "")
            model_name = None
            
            # Try to query the pod directly to get actual model name
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"http://{pod_ip}:{SERVICE_PORT}/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_name = data["data"][0].get("id", "")
                            logger.info(f"Queried pod {pod.metadata.name} ({pod_ip}): found model '{model_name}'")
            except Exception as e:
                logger.warning(f"Failed to query pod {pod.metadata.name} ({pod_ip}): {e}")
            
            # Fallback: try to extract from pod name, service name, or labels
            if not model_name:
                pod_name = pod.metadata.name
                labels = pod.metadata.labels or {}
                
                # Try to extract from pod name (e.g., ms-vllm-qwen2-5-0-5b-... -> Qwen/Qwen2.5-0.5B-Instruct)
                if "qwen2-5-0-5b" in pod_name.lower() or "qwen2.5-0.5b" in pod_name.lower():
                    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                elif "llama-32-1b" in pod_name.lower() or "llama-3.2-1b" in pod_name.lower():
                    model_name = "meta-llama/Llama-3.2-1B-Instruct"
                
                # Try to extract from service name
                if not model_name and service_name:
                    if "qwen2-5-0-5b" in service_name.lower() or "qwen2.5-0.5b" in service_name.lower():
                        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                    elif "llama-32-1b" in service_name.lower() or "llama-3.2-1b" in service_name.lower():
                        model_name = "meta-llama/Llama-3.2-1B-Instruct"
            
            # Determine owned_by from service name, pod name, or labels
            owned_by = None
            pod_name = pod.metadata.name
            if "vllm" in pod_name.lower() or (service_name and "vllm" in service_name.lower()):
                owned_by = "vllm"
            elif "sglang" in pod_name.lower() or (service_name and "sglang" in service_name.lower()):
                owned_by = "sglang"
            
            # Fallback to labels
            if not owned_by:
                labels = pod.metadata.labels or {}
                owned_by = labels.get("llm-d.ai/owned-by") or labels.get("app", "vllm")
            
            # Default to vllm if not specified
            if not owned_by or owned_by == "":
                owned_by = "vllm"
            
            # Return discovered mapping
            result = {}
            # Use model name from query, or fallback to extracted name
            if not model_name:
                # Final fallback: use default model name based on pod name
                if "qwen" in pod_name.lower():
                    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                elif "llama" in pod_name.lower():
                    model_name = "meta-llama/Llama-3.2-1B-Instruct"
            
            if model_name:
                POD_IP_CACHE[model_name] = pod_ip
                
                key = (model_name, owned_by)
                # For ModelService, prefer InferencePool Gateway service
                # Find the corresponding InferencePool Gateway service based on owned_by
                target_service = None
                
                # Look for InferencePool Gateway service matching this model and engine
                # Extract release name from pod name (e.g., ms-vllm-qwen2-5-0-5b-... -> vllm-qwen2-5-0-5b)
                gateway_service_name = None
                if "ms-vllm" in pod_name or "vllm" in pod_name.lower():
                    # Try to extract release name from pod name
                    if "vllm-qwen2-5-0-5b" in pod_name:
                        gateway_service_name = "infra-vllm-qwen2-5-0-5b-inference-gateway-istio"
                    elif owned_by == "vllm":
                        # Generic fallback
                        gateway_service_name = f"infra-vllm-qwen2-5-0-5b-inference-gateway-istio"
                elif "ms-sglang" in pod_name or "sglang" in pod_name.lower():
                    if "sglang-qwen2-5-0-5b" in pod_name:
                        gateway_service_name = "infra-sglang-qwen2-5-0-5b-inference-gateway-istio"
                    elif owned_by == "sglang":
                        gateway_service_name = f"infra-sglang-qwen2-5-0-5b-inference-gateway-istio"
                
                # Check if this gateway service exists - no fallback, fast fail
                if gateway_service_name:
                    try:
                        svc = v1.read_namespaced_service(gateway_service_name, NAMESPACE)
                        target_service = gateway_service_name
                        logger.info(f"Using InferencePool Gateway service: {target_service} for {model_name} ({owned_by})")
                    except Exception as e:
                        logger.error(f"Gateway service {gateway_service_name} not found: {e}")
                        # No fallback - fast fail
                        target_service = None
                else:
                    target_service = None
                
                # If no gateway service found, don't use fallback - fail fast
                if not target_service:
                    logger.warning(f"No gateway service found for {model_name} ({owned_by}), skipping discovery")
                    return result  # Return empty result instead of continue (not in a loop)
                
                result[key] = target_service
                logger.info(f"Discovered: {model_name} (owned_by: {owned_by}) -> '{target_service}' (pod: {pod.metadata.name})")
            return result
        
        # Query all pods in parallel
        tasks = []
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            if not pod_ip:
                continue
            tasks.append(query_single_pod(pod, pod_ip, pod_to_service))
        
        # Run all queries in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict) and result:
                    discovered.update(result)
        
        # Update routing config with discovered services
        ROUTING_CONFIG.update(discovered)
        return discovered
    except Exception as e:
        logger.warning(f"Failed to discover services from Kubernetes: {e}")
        return {}

# Pydantic models
class RoutingMapping(BaseModel):
    model: str
    owned_by: Optional[str] = None
    service_name: str

class RoutingMappingResponse(BaseModel):
    model: str
    owned_by: Optional[str]
    service_name: str

def get_service_for_model_and_owned_by(model: str, owned_by: Optional[str] = None) -> Optional[str]:
    """
    Get corresponding Service name based on model name and owned_by
    
    Args:
        model: Model name from request
        owned_by: Inference engine type (vllm, sglang, etc.) from 'owned_by' field or header
    
    Returns:
        Service name or None if not found
    """
    # Normalize owned_by
    if owned_by:
        owned_by = owned_by.lower()
    
    # Try exact match with owned_by
    if owned_by:
        key = (model, owned_by)
        if key in ROUTING_CONFIG:
            service = ROUTING_CONFIG[key]
            if service:
                logger.info(f"Matched model '{model}' with owned_by '{owned_by}' to service '{service}'")
                return service
    
    # Try match with None (default/any owned_by)
    key = (model, None)
    if key in ROUTING_CONFIG:
        service = ROUTING_CONFIG[key]
        if service:
            logger.info(f"Matched model '{model}' (default owned_by) to service '{service}'")
            return service
    
    # Try default vllm
    key = (model, "vllm")
    if key in ROUTING_CONFIG:
        service = ROUTING_CONFIG[key]
        if service:
            logger.info(f"Matched model '{model}' (default vllm) to service '{service}'")
            return service
    
    # Fuzzy match - try to find similar model names
    model_lower = model.lower()
    for (key_model, key_owned_by), service in ROUTING_CONFIG.items():
        if service is None:
            continue
        
        key_model_lower = key_model.lower() if key_model else ""
        if key_model_lower in model_lower or model_lower in key_model_lower:
            if (key_owned_by is None) or (owned_by and key_owned_by == owned_by.lower()):
                logger.info(f"Matched model '{model}' (fuzzy) with owned_by '{owned_by}' to service '{service}'")
                return service
    
    return None

async def forward_request(
    service_name: str,
    path: str,
    method: str,
    headers: Dict[str, str],
    body: bytes = None,
    params: Dict[str, Any] = None,
    model: str = None
) -> httpx.Response:
    """Forward request to Kubernetes service"""
    target_url = None
    
    # Check if service_name is a pod IP
    is_pod_ip = (service_name.replace(".", "").replace(":", "").isdigit() or ":" in service_name) and not any(c.isalpha() for c in service_name)
    
    if is_pod_ip:
        target_url = f"http://{service_name}:{SERVICE_PORT}{path}"
        logger.info(f"Using pod IP directly (fallback): {service_name}")
    else:
        # Regular Kubernetes service, use service DNS
        # Check if this is an InferencePool Gateway service (uses port 80)
        if "inference-gateway-istio" in service_name:
            gateway_port = 80
            target_url = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:{gateway_port}{path}"
            logger.info(f"Using InferencePool Gateway service: {service_name} on port {gateway_port}")
        else:
            target_url = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:{SERVICE_PORT}{path}"
            logger.info(f"Using Kubernetes service: {service_name} on port {SERVICE_PORT}")
    
    # Remove headers that shouldn't be forwarded
    forward_headers = {k: v for k, v in headers.items() 
                      if k.lower() not in ['host', 'content-length']}
    
    logger.info(f"Forwarding {method} {path} to {target_url}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        if method == "GET":
            response = await client.get(target_url, headers=forward_headers, params=params)
        elif method == "POST":
            response = await client.post(
                target_url,
                headers=forward_headers,
                content=body,
                params=params
            )
        elif method == "DELETE":
            response = await client.delete(target_url, headers=forward_headers)
        else:
            raise HTTPException(status_code=405, detail=f"Method {method} not supported")
        
        return response

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "engine-comparison-gateway", "namespace": NAMESPACE}

@app.post("/admin/discover")
async def rediscover_services():
    """Manually trigger service discovery"""
    discovered = await discover_services()
    services_dict = {f"{model}::{owned_by or 'default'}": service for (model, owned_by), service in discovered.items()}
    return {"message": "Service discovery completed", "discovered": len(discovered), "services": services_dict}

@app.get("/admin/api/routing", response_model=List[RoutingMappingResponse])
async def get_routing_config():
    """Get all routing mappings"""
    mappings = []
    for (model, owned_by), service_name in ROUTING_CONFIG.items():
        display_name = POD_IP_CACHE.get(model, service_name)
        mappings.append(RoutingMappingResponse(
            model=model,
            owned_by=owned_by,
            service_name=display_name
        ))
    return mappings

@app.post("/admin/api/routing", response_model=RoutingMappingResponse)
async def add_routing_mapping(mapping: RoutingMapping):
    """Add a new routing mapping"""
    key = (mapping.model, mapping.owned_by)
    if key in ROUTING_CONFIG:
        raise HTTPException(
            status_code=400, 
            detail=f"Mapping already exists for model '{mapping.model}' with owned_by '{mapping.owned_by}'"
        )
    
    ROUTING_CONFIG[key] = mapping.service_name
    logger.info(f"Added routing mapping: {key} -> {mapping.service_name}")
    
    return RoutingMappingResponse(
        model=mapping.model,
        owned_by=mapping.owned_by,
        service_name=mapping.service_name
    )

@app.delete("/admin/api/routing")
async def delete_routing_mapping(model: str, owned_by: Optional[str] = None):
    """Delete a routing mapping"""
    if owned_by == "":
        owned_by = None
    
    key = (model, owned_by)
    if key not in ROUTING_CONFIG:
        raise HTTPException(
            status_code=404, 
            detail=f"Mapping not found for model '{model}' with owned_by '{owned_by}'"
        )
    
    del ROUTING_CONFIG[key]
    logger.info(f"Deleted routing mapping: {key}")
    
    return {"message": "Mapping deleted successfully"}

@app.get("/v1/models")
async def list_models():
    """List all available models from all LLMInferenceServices"""
    models = []
    seen_keys = set()
    pod_model_cache = {}
    
    # Query pods directly to get complete model information
    try:
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector="llm-d.ai/role=decode"
        )
        for pod in pods.items:
            pod_ip = pod.status.pod_ip
            if not pod_ip:
                continue
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"http://{pod_ip}:{SERVICE_PORT}/v1/models")
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            pod_model_cache[pod_ip] = data["data"][0]
                            logger.info(f"Retrieved model from pod {pod.metadata.name} ({pod_ip}): {data['data'][0].get('id')}")
            except Exception as e:
                logger.warning(f"Failed to query pod {pod.metadata.name} ({pod_ip}): {e}")
    except Exception as e:
        logger.warning(f"Failed to query pods directly: {e}")
    
    # Build models list from routing config and pod cache
    service_to_mappings = {}
    for (model_name, owned_by), service_name in ROUTING_CONFIG.items():
        if service_name and service_name is not None:
            if service_name not in service_to_mappings:
                service_to_mappings[service_name] = []
            service_to_mappings[service_name].append((model_name, owned_by))
    
    # For each service, get models and match with pod cache
    for service_name, mappings in service_to_mappings.items():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                target_url = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:{SERVICE_PORT}/v1/models"
                response = await client.get(target_url)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        for model in data["data"]:
                            model_id = model.get("id", "")
                            if model_id:
                                for model_name, owned_by in mappings:
                                    if model_id == model_name or model_name in model_id or model_id in model_name:
                                        key = (model_id, owned_by)
                                        if key not in seen_keys:
                                            seen_keys.add(key)
                                            engine_name = owned_by or "vllm"
                                            
                                            # Try to find in pod cache for complete metadata
                                            model_entry = None
                                            for pod_ip, pod_model in pod_model_cache.items():
                                                pod_model_id = pod_model.get("id", "")
                                                if pod_model_id == model_id:
                                                    model_entry = pod_model.copy()
                                                    break
                                            
                                            if model_entry is None:
                                                model_entry = model.copy()
                                            
                                            model_entry["id"] = model_id
                                            model_entry.pop("permission", None)
                                            model_entry.pop("engine", None)
                                            model_entry["owned_by"] = engine_name
                                            models.append(model_entry)
                                            logger.info(f"Added model {model_id} (owned_by: {engine_name})")
        except Exception as e:
            logger.warning(f"Failed to get models from {service_name}: {e}")
    
    # Add models from pod cache that weren't matched
    # This ensures we include all models even if InferencePool Gateway services are unavailable
    for pod_ip, pod_model in pod_model_cache.items():
        pod_model_id = pod_model.get("id", "")
        if pod_model_id:
            # Determine owned_by from pod name or ROUTING_CONFIG
            owned_by = None
            
            # Try to find the pod to get its name for owned_by detection
            try:
                pods = v1.list_namespaced_pod(
                    namespace=NAMESPACE,
                    label_selector="llm-d.ai/role=decode"
                )
                for pod in pods.items:
                    if pod.status.pod_ip == pod_ip:
                        pod_name = pod.metadata.name
                        # Extract owned_by from pod name
                        if "vllm" in pod_name.lower():
                            owned_by = "vllm"
                        elif "sglang" in pod_name.lower():
                            owned_by = "sglang"
                        break
            except Exception as e:
                logger.warning(f"Failed to get pod info for IP {pod_ip}: {e}")
            
            # Fallback to ROUTING_CONFIG
            if not owned_by:
                for (config_model, config_owned_by), config_svc in ROUTING_CONFIG.items():
                    if (config_model == pod_model_id or 
                        pod_model_id.lower() in config_model.lower() or 
                        config_model.lower() in pod_model_id.lower()):
                        owned_by = config_owned_by or "vllm"
                        break
            
            # Final fallback
            if not owned_by:
                owned_by = "vllm"
            
            # Check if this specific (model_id, owned_by) combination is already added
            key = (pod_model_id, owned_by)
            if key not in seen_keys:
                seen_keys.add(key)
                model_entry = pod_model.copy()
                model_entry["id"] = pod_model_id
                model_entry.pop("permission", None)
                model_entry.pop("engine", None)
                model_entry["owned_by"] = owned_by
                models.append(model_entry)
                logger.info(f"Added model from pod cache: {pod_model_id} (owned_by: {owned_by})")
    
    logger.info(f"Returning {len(models)} models total")
    return {"object": "list", "data": models}

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_v1(request: Request, path: str):
    """Proxy all /v1/* requests (except /v1/models which is handled above)"""
    if path == "models" and request.method == "GET":
        return await list_models()
    
    body = await request.body()
    
    model = None
    owned_by = None
    
    if body:
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
            owned_by = (body_json.get("owned_by") or 
                       body_json.get("engine") or 
                       body_json.get("inference_server"))
        except json.JSONDecodeError:
            pass
    
    if not model:
        model = request.query_params.get("model")
    
    if not owned_by:
        owned_by = (request.headers.get("x-owned-by") or
                   request.headers.get("X-Owned-By") or
                   request.query_params.get("owned_by") or
                   request.query_params.get("engine") or
                   request.query_params.get("inference_server"))
    
    if not model:
        raise HTTPException(
            status_code=400,
            detail="Missing 'model' field in request body, query parameters, or headers"
        )
    
    service_name = get_service_for_model_and_owned_by(model, owned_by)
    if not service_name:
        if not ROUTING_CONFIG:
            logger.info("Routing config is empty, triggering service discovery...")
            await discover_services()
        service_name = get_service_for_model_and_owned_by(model, owned_by)
    
    if not service_name:
        available_models = [f"{m} (owned_by: {o})" if o else m 
                           for (m, o) in ROUTING_CONFIG.keys() if m]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' with owned_by '{owned_by}' not found. "
                   f"Available: {available_models}. "
                   f"Try calling POST /admin/discover to refresh service discovery."
        )
    
    logger.info(f"Routing request for model '{model}' with owned_by '{owned_by}' to service '{service_name}'")
    
    try:
        response = await forward_request(
            service_name=service_name,
            path=f"/v1/{path}",
            method=request.method,
            headers=dict(request.headers),
            body=body,
            params=dict(request.query_params),
            model=model
        )
        
        if "text/event-stream" in response.headers.get("content-type", ""):
            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        return JSONResponse(
            content=response.json() if response.content else {},
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gateway timeout")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Service '{service_name}' is not available. Please check if the pod is running."
        )
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
