#!/bin/bash

# PDRM Asset Management Dashboard - Docker Build and Push Script
# This script builds the Docker image and pushes it to Docker Hub

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DOCKERHUB_USERNAME=""
IMAGE_NAME="pdrm-dashboard"
TAG="latest"
SKIP_PUSH=false

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

print_color $BLUE "🐳 PDRM Dashboard - Docker Build & Push Script"
print_color $BLUE "============================================="

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            DOCKERHUB_USERNAME="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --skip-push)
            SKIP_PUSH=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -u, --username DOCKERHUB_USERNAME   Docker Hub username"
            echo "  -i, --image IMAGE_NAME             Image name (default: pdrm-dashboard)"
            echo "  -t, --tag TAG                      Image tag (default: latest)"
            echo "  --skip-push                        Skip pushing to Docker Hub"
            echo "  -h, --help                         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Get Docker Hub username if not provided
if [ -z "$DOCKERHUB_USERNAME" ]; then
    read -p "Enter your Docker Hub username: " DOCKERHUB_USERNAME
fi

# Validate inputs
if [ -z "$DOCKERHUB_USERNAME" ]; then
    print_color $RED "❌ Docker Hub username is required!"
    exit 1
fi

FULL_IMAGE_NAME="$DOCKERHUB_USERNAME/$IMAGE_NAME:$TAG"

print_color $YELLOW "📋 Build Configuration:"
print_color $YELLOW "  Docker Hub Username: $DOCKERHUB_USERNAME"
print_color $YELLOW "  Image Name: $IMAGE_NAME"
print_color $YELLOW "  Tag: $TAG"
print_color $YELLOW "  Full Image Name: $FULL_IMAGE_NAME"
echo

# Check if Docker is running
print_color $BLUE "🔍 Checking Docker status..."
if ! docker ps >/dev/null 2>&1; then
    print_color $RED "❌ Docker is not running. Please start Docker."
    exit 1
fi
print_color $GREEN "✅ Docker is running"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_color $RED "❌ Dockerfile not found in current directory!"
    exit 1
fi
print_color $GREEN "✅ Dockerfile found"

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    print_color $RED "❌ requirements.txt not found!"
    exit 1
fi
print_color $GREEN "✅ requirements.txt found"

# Check for main application file
if [ ! -f "streamlit_dashboard.py" ]; then
    print_color $RED "❌ streamlit_dashboard.py not found!"
    exit 1
fi
print_color $GREEN "✅ Main application file found"

# Build the Docker image
print_color $BLUE "🏗️  Building Docker image..."
print_color $YELLOW "Command: docker build -t $FULL_IMAGE_NAME ."

if docker build -t "$FULL_IMAGE_NAME" .; then
    print_color $GREEN "✅ Docker image built successfully!"
else
    print_color $RED "❌ Docker build failed!"
    exit 1
fi

# Display image info
print_color $BLUE "📊 Image Information:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

if [ "$SKIP_PUSH" = false ]; then
    # Login to Docker Hub
    print_color $BLUE "🔐 Please login to Docker Hub..."
    if docker login; then
        print_color $GREEN "✅ Successfully logged in to Docker Hub"
    else
        print_color $RED "❌ Docker Hub login failed!"
        exit 1
    fi
    
    # Push to Docker Hub
    print_color $BLUE "🚀 Pushing image to Docker Hub..."
    print_color $YELLOW "Command: docker push $FULL_IMAGE_NAME"
    
    if docker push "$FULL_IMAGE_NAME"; then
        print_color $GREEN "✅ Image pushed successfully to Docker Hub!"
        print_color $GREEN "🎉 Your image is now available at: https://hub.docker.com/r/$DOCKERHUB_USERNAME/$IMAGE_NAME"
    else
        print_color $RED "❌ Docker push failed!"
        exit 1
    fi
else
    print_color $YELLOW "⏭️  Skipping push to Docker Hub (--skip-push flag used)"
fi

echo
print_color $BLUE "🎯 Quick Usage Commands:"
print_color $YELLOW "  Run locally: docker run -p 8501:8501 $FULL_IMAGE_NAME"
print_color $YELLOW "  Pull from hub: docker pull $FULL_IMAGE_NAME"
echo
print_color $GREEN "✨ Build and push completed successfully!"
