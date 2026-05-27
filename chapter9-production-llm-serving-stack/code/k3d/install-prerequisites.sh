#!/bin/bash
set -e

echo "Installing prerequisites for k3d GPU cluster setup..."

# 1. Check NVIDIA Drivers
echo "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi
nvidia-smi
echo "✓ NVIDIA drivers installed"

# 2. Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker first."
    exit 1
fi
docker --version
echo "✓ Docker installed"

# 3. Install NVIDIA Container Toolkit (if not already installed)
echo "Checking NVIDIA Container Toolkit..."
if command -v nvidia-ctk &> /dev/null; then
    echo "NVIDIA Container Toolkit already installed"
    nvidia-ctk --version
else
    echo "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    nvidia-ctk --version
fi

# Configure Docker runtime (only if not already configured)
if grep -q "nvidia" /etc/docker/daemon.json 2>/dev/null; then
    echo "Docker already configured for NVIDIA runtime"
else
    echo "Configuring Docker runtime for NVIDIA..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi
echo "✓ NVIDIA Container Toolkit configured"

# 4. Install kubectl
echo "Checking kubectl..."
if command -v kubectl &> /dev/null; then
    echo "kubectl already installed"
    kubectl version --client --short 2>/dev/null || kubectl version --client
else
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    rm kubectl
    kubectl version --client
fi
echo "✓ kubectl ready"

# 5. Install k3d
echo "Checking k3d..."
if command -v k3d &> /dev/null; then
    echo "k3d already installed"
    k3d --version
else
    echo "Installing k3d..."
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
    k3d --version
fi
echo "✓ k3d ready"

echo ""
echo "Prerequisites installation complete!"
echo ""
echo "Next steps:"
echo "1. Build the custom k3s-cuda image: ./build.sh"
echo "2. Create the cluster: ./create-cluster.sh"
