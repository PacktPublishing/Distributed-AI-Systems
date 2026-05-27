#!/bin/bash
# Script to clean up LLM resources in default namespace
# These resources were deployed before namespace isolation was implemented
# After cleanup, use manage-cluster-multi-engines.sh or manage-cluster-multi-models.sh
# which deploy to dedicated namespaces (multi-engines or multi-models)

set -e

NAMESPACE="${NAMESPACE:-default}"

echo "=========================================="
echo "Cleaning up LLM resources in namespace: $NAMESPACE"
echo "=========================================="
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "⚠️  Namespace '$NAMESPACE' does not exist"
    exit 1
fi

# List resources to be deleted
echo "📋 Resources to be deleted:"
echo ""

echo "Pods:"
kubectl get pods -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" || echo "  (none)"
echo ""

echo "Deployments:"
kubectl get deployments -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" || echo "  (none)"
echo ""

echo "Services:"
kubectl get svc -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" || echo "  (none)"
echo ""

# Confirm deletion
read -p "⚠️  Do you want to delete these resources? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "🗑️  Deleting resources..."

# Delete Pods
echo "  Deleting Pods..."
kubectl get pods -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" | while read -r pod; do
    if [ -n "$pod" ]; then
        echo "    Deleting $pod..."
        kubectl delete "$pod" -n "$NAMESPACE" 2>/dev/null || true
    fi
done

# Delete Deployments
echo "  Deleting Deployments..."
kubectl get deployments -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" | while read -r deployment; do
    if [ -n "$deployment" ]; then
        echo "    Deleting $deployment..."
        kubectl delete "$deployment" -n "$NAMESPACE" 2>/dev/null || true
    fi
done

# Delete Services
echo "  Deleting Services..."
kubectl get svc -n "$NAMESPACE" -o name 2>/dev/null | grep -E "vllm|sglang" | while read -r service; do
    if [ -n "$service" ]; then
        echo "    Deleting $service..."
        kubectl delete "$service" -n "$NAMESPACE" 2>/dev/null || true
    fi
done

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📊 Remaining resources in $NAMESPACE namespace:"
kubectl get pods,deployments,svc -n "$NAMESPACE" 2>/dev/null | grep -E "vllm|sglang" || echo "  (none)"
echo ""
echo "💡 To deploy resources in isolated namespaces, use:"
echo "   ./manage-cluster-multi-engines.sh start   # Deploys to 'multi-engines' namespace"
echo "   ./manage-cluster-multi-models.sh start    # Deploys to 'multi-models' namespace"
