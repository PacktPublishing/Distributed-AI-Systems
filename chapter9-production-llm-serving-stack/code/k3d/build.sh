#!/bin/bash
set -e

# Auto-detect CUDA version from local installation
detect_cuda_version() {
    # Try version.json first (CUDA 11.1+), it has the full version
    if [ -f /usr/local/cuda/version.json ]; then
        local version=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)".*/\1/p' /usr/local/cuda/version.json | head -1)
        if [ -n "$version" ]; then
            echo "$version"
            return
        fi
    fi
    # Try nvcc - note it only gives major.minor (e.g., 12.4), we need to append .1
    if command -v nvcc &> /dev/null; then
        local major_minor=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        if [ -n "$major_minor" ]; then
            echo "${major_minor}.1"  # Append .1 as common patch version
            return
        fi
    fi
    echo "12.4.1"  # Default fallback
}

# Get latest stable k3s version from GitHub API
get_latest_k3s_tag() {
    local latest=$(curl -s --connect-timeout 5 "https://api.github.com/repos/k3s-io/k3s/releases/latest" 2>/dev/null | \
                   grep '"tag_name"' | sed 's/.*"tag_name": "\([^"]*\)".*/\1/' | head -1)
    if [ -n "$latest" ]; then
        echo "$latest"
    else
        echo "v1.35.1+k3s1"  # Fallback
    fi
}

# Check prerequisites
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found. Run this script from the k3d directory."
    exit 1
fi

if [ ! -f "device-plugin-daemonset.yaml" ]; then
    echo "ERROR: device-plugin-daemonset.yaml not found. Run this script from the k3d directory."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "Detecting versions..."

# Auto-detect or use environment variables
if [ -z "$CUDA_TAG" ]; then
    CUDA_VERSION=$(detect_cuda_version)
    CUDA_TAG="${CUDA_VERSION}-base-ubuntu22.04"
    echo "Auto-detected CUDA version: $CUDA_VERSION"
fi

if [ -z "$K3S_TAG" ]; then
    K3S_TAG=$(get_latest_k3s_tag | sed 's/+/-/')
    echo "Latest k3s version: $K3S_TAG"
fi

# Auto-generate image name from tags if not specified
if [ -z "$IMAGE_NAME" ]; then
    K3S_VERSION=$(echo "$K3S_TAG" | sed 's/^v//' | cut -d'-' -f1)
    CUDA_VERSION=$(echo "$CUDA_TAG" | cut -d'-' -f1)
    IMAGE_NAME="k3s-cuda:${K3S_VERSION}-cuda-${CUDA_VERSION}"
fi

echo ""
echo "Building custom k3s-cuda image..."
echo "K3S_TAG: $K3S_TAG"
echo "CUDA_TAG: $CUDA_TAG"
echo "IMAGE_NAME: $IMAGE_NAME"
echo ""
echo "To override versions:"
echo "  K3S_TAG=v1.32.0-k3s1 CUDA_TAG=13.0.0-base-ubuntu24.04 ./build.sh"
echo ""

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Ask for confirmation (skip with -y flag)
if [ "$1" != "-y" ]; then
    read -p "Proceed with build? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Build cancelled."
        exit 0
    fi
fi

# Check if buildx is available
if docker buildx version &> /dev/null; then
    echo "Using docker buildx..."
    docker buildx build \
      --build-arg K3S_TAG="$K3S_TAG" \
      --build-arg CUDA_TAG="$CUDA_TAG" \
      -t "$IMAGE_NAME" \
      --load .
else
    echo "Using standard docker build..."
    docker build \
      --build-arg K3S_TAG="$K3S_TAG" \
      --build-arg CUDA_TAG="$CUDA_TAG" \
      -t "$IMAGE_NAME" \
      .
fi

echo ""
echo "✓ Image built successfully: $IMAGE_NAME"
echo ""
echo "Verify the image:"
echo "  docker images | grep k3s-cuda"
echo ""
echo "Next step: Create the cluster with ./create-cluster.sh"
