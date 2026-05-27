#!/bin/bash
# Test Gateway using cluster-internal access
export NAMESPACE=llm-d-multi-engine

echo "=== Testing Custom API Gateway ==="
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} engine-comparison-gateway -o jsonpath='{.spec.clusterIP}')
echo "Gateway IP: $GATEWAY_IP"
echo ""

echo "1. List Models:"
kubectl run -n ${NAMESPACE} --rm -i --restart=Never test-models --image=curlimages/curl:latest -- \
  curl -s http://${GATEWAY_IP}:8000/v1/models | python3 -m json.tool
echo ""

echo "2. Health Check:"
kubectl run -n ${NAMESPACE} --rm -i --restart=Never test-health --image=curlimages/curl:latest -- \
  curl -s http://${GATEWAY_IP}:8000/health | python3 -m json.tool
