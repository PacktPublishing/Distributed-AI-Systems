# Quick Test Guide

## Problem: "Empty reply from server" with port-forward

If you get this error when using `curl http://localhost:9000/v1/models`, try these solutions:

### Solution 1: Clean port and retry

```bash
# Kill any existing port-forward
lsof -ti:9000 | xargs kill -9 2>/dev/null

# Start fresh port-forward
export NAMESPACE=llm-d-multi-engine
GATEWAY_POD=$(kubectl get pods -n ${NAMESPACE} -l app=engine-comparison-gateway | grep Running | tail -1 | awk '{print $1}')
kubectl port-forward -n ${NAMESPACE} ${GATEWAY_POD} 9000:8000

# Wait 3-5 seconds, then in another terminal:
curl http://localhost:9000/v1/models
```

### Solution 2: Use cluster-internal access (Recommended)

```bash
export NAMESPACE=llm-d-multi-engine
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} engine-comparison-gateway -o jsonpath='{.spec.clusterIP}')

# List models
kubectl run -n ${NAMESPACE} --rm -i --restart=Never test-models --image=curlimages/curl:latest -- \
  curl -s http://${GATEWAY_IP}:8000/v1/models

# Health check
kubectl run -n ${NAMESPACE} --rm -i --restart=Never test-health --image=curlimages/curl:latest -- \
  curl -s http://${GATEWAY_IP}:8000/health
```

### Solution 3: Use service port-forward

```bash
kubectl port-forward -n llm-d-multi-engine svc/engine-comparison-gateway 9000:8000
```
