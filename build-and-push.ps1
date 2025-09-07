#!/usr/bin/env pwsh

# PDRM Asset Management Dashboard - Docker Build and Push Script
# This script builds the Docker image and pushes it to Docker Hub

param(
    [Parameter(Mandatory=$false)]
    [string]$DockerHubUsername = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageName = "pdrm-dashboard",
    
    [Parameter(Mandatory=$false)]
    [string]$Tag = "latest",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipPush = $false
)

# Colors for output
$Green = "`e[32m"
$Red = "`e[31m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

# Function to write colored output
function Write-ColorOutput {
    param([string]$Message, [string]$Color)
    Write-Host "$Color$Message$Reset"
}

Write-ColorOutput "🐳 PDRM Dashboard - Docker Build & Push Script" $Blue
Write-ColorOutput "=============================================" $Blue

# Get Docker Hub username if not provided
if ([string]::IsNullOrWhiteSpace($DockerHubUsername)) {
    $DockerHubUsername = Read-Host "Enter your Docker Hub username"
}

# Validate inputs
if ([string]::IsNullOrWhiteSpace($DockerHubUsername)) {
    Write-ColorOutput "❌ Docker Hub username is required!" $Red
    exit 1
}

$FullImageName = "$DockerHubUsername/$ImageName`:$Tag"

Write-ColorOutput "📋 Build Configuration:" $Yellow
Write-ColorOutput "  Docker Hub Username: $DockerHubUsername" $Yellow
Write-ColorOutput "  Image Name: $ImageName" $Yellow
Write-ColorOutput "  Tag: $Tag" $Yellow
Write-ColorOutput "  Full Image Name: $FullImageName" $Yellow
Write-ColorOutput "" $Yellow

# Check if Docker is running
Write-ColorOutput "🔍 Checking Docker status..." $Blue
try {
    docker ps | Out-Null
    Write-ColorOutput "✅ Docker is running" $Green
} catch {
    Write-ColorOutput "❌ Docker is not running. Please start Docker Desktop." $Red
    exit 1
}

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile")) {
    Write-ColorOutput "❌ Dockerfile not found in current directory!" $Red
    exit 1
}

Write-ColorOutput "✅ Dockerfile found" $Green

# Check for requirements.txt
if (-not (Test-Path "requirements.txt")) {
    Write-ColorOutput "❌ requirements.txt not found!" $Red
    exit 1
}

Write-ColorOutput "✅ requirements.txt found" $Green

# Check for main application file
if (-not (Test-Path "streamlit_dashboard.py")) {
    Write-ColorOutput "❌ streamlit_dashboard.py not found!" $Red
    exit 1
}

Write-ColorOutput "✅ Main application file found" $Green

# Build the Docker image
Write-ColorOutput "🏗️  Building Docker image..." $Blue
Write-ColorOutput "Command: docker build -t $FullImageName ." $Yellow

$BuildResult = docker build -t $FullImageName . 2>&1
$BuildExitCode = $LASTEXITCODE

if ($BuildExitCode -eq 0) {
    Write-ColorOutput "✅ Docker image built successfully!" $Green
} else {
    Write-ColorOutput "❌ Docker build failed!" $Red
    Write-ColorOutput "Build output:" $Red
    $BuildResult | ForEach-Object { Write-ColorOutput "  $_" $Red }
    exit 1
}

# Display image info
Write-ColorOutput "📊 Image Information:" $Blue
docker images $FullImageName --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

if (-not $SkipPush) {
    # Login to Docker Hub
    Write-ColorOutput "🔐 Please login to Docker Hub..." $Blue
    $LoginResult = docker login 2>&1
    $LoginExitCode = $LASTEXITCODE
    
    if ($LoginExitCode -eq 0) {
        Write-ColorOutput "✅ Successfully logged in to Docker Hub" $Green
    } else {
        Write-ColorOutput "❌ Docker Hub login failed!" $Red
        Write-ColorOutput "Login output:" $Red
        $LoginResult | ForEach-Object { Write-ColorOutput "  $_" $Red }
        exit 1
    }
    
    # Push to Docker Hub
    Write-ColorOutput "🚀 Pushing image to Docker Hub..." $Blue
    Write-ColorOutput "Command: docker push $FullImageName" $Yellow
    
    $PushResult = docker push $FullImageName 2>&1
    $PushExitCode = $LASTEXITCODE
    
    if ($PushExitCode -eq 0) {
        Write-ColorOutput "✅ Image pushed successfully to Docker Hub!" $Green
        Write-ColorOutput "🎉 Your image is now available at: https://hub.docker.com/r/$DockerHubUsername/$ImageName" $Green
    } else {
        Write-ColorOutput "❌ Docker push failed!" $Red
        Write-ColorOutput "Push output:" $Red
        $PushResult | ForEach-Object { Write-ColorOutput "  $_" $Red }
        exit 1
    }
} else {
    Write-ColorOutput "⏭️  Skipping push to Docker Hub (--SkipPush flag used)" $Yellow
}

Write-ColorOutput "" $Blue
Write-ColorOutput "🎯 Quick Usage Commands:" $Blue
Write-ColorOutput "  Run locally: docker run -p 8501:8501 $FullImageName" $Yellow
Write-ColorOutput "  Pull from hub: docker pull $FullImageName" $Yellow
Write-ColorOutput "" $Blue
Write-ColorOutput "✨ Build and push completed successfully!" $Green
