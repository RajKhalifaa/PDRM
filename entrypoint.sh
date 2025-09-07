#!/bin/bash

# PDRM Asset Management Dashboard - Entrypoint Script
# This script starts the Streamlit application

set -e

echo "🚀 Starting PDRM Asset Management Dashboard..."
echo "📊 Streamlit server starting on port 8501..."

# Start virtual display for plotly chart generation
echo "🖥️  Starting virtual display..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Wait a moment for Xvfb to start
sleep 2

# Set environment variables
export DISPLAY=:99
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

echo "🔧 Environment configured:"
echo "  - Port: $STREAMLIT_SERVER_PORT"
echo "  - Address: $STREAMLIT_SERVER_ADDRESS" 
echo "  - Display: $DISPLAY"

# Check if required files exist
if [ ! -f "/app/streamlit_dashboard.py" ]; then
    echo "❌ Main application file not found!"
    exit 1
fi

if [ ! -f "/app/requirements.txt" ]; then
    echo "❌ Requirements file not found!"
    exit 1
fi

echo "✅ Application files verified"

# Create logs directory
mkdir -p /app/logs

# Start the Streamlit application
echo "🎯 Launching Streamlit application..."
cd /app

exec streamlit run streamlit_dashboard.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none \
    --server.maxUploadSize=200 \
    --global.developmentMode=false
