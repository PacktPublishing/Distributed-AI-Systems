#!/bin/bash
# Script to clean up Pods in error states (UnexpectedAdmissionError, Failed, etc.)
# Usage: ./cleanup-error-pods.sh [namespace]
# If namespace not specified, cleans up all namespaces

NAMESPACE="${1:-}"

if [ -z "$NAMESPACE" ]; then
    echo "🧹 Cleaning up error Pods in all namespaces..."
    NAMESPACES=$(kubectl get namespaces -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -E "multi-engines|multi-models|default" || echo "")
    for ns in $NAMESPACES; do
        echo ""
        echo "Namespace: $ns"
        ERROR_PODS=$(kubectl get pods -n "$ns" -o jsonpath='{range .items[?(@.status.phase=="Failed" || @.status.reason=="UnexpectedAdmissionError")]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
        if [ -n "$ERROR_PODS" ]; then
            echo "$ERROR_PODS" | while read -r pod; do
                if [ -n "$pod" ]; then
                    echo "  🗑️  Deleting: $pod"
                    kubectl delete pod "$pod" -n "$ns" --grace-period=0 --force 2>/dev/null || true
                fi
            done
        else
            echo "  ✅ No error Pods found"
        fi
    done
else
    echo "🧹 Cleaning up error Pods in namespace: $NAMESPACE..."
    ERROR_PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[?(@.status.phase=="Failed" || @.status.reason=="UnexpectedAdmissionError")]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    if [ -n "$ERROR_PODS" ]; then
        echo "$ERROR_PODS" | while read -r pod; do
            if [ -n "$pod" ]; then
                echo "  🗑️  Deleting: $pod"
                kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=0 --force 2>/dev/null || true
            fi
        done
    else
        echo "  ✅ No error Pods found"
    fi
fi

echo ""
echo "✅ Cleanup completed!"
