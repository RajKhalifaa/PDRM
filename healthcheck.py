#!/usr/bin/env python3

"""
PDRM Asset Management Dashboard - Health Check Script
This script checks if the Streamlit application is running and responding.
"""

import requests
import sys
import os

def check_health():
    """Check if the Streamlit application is healthy."""
    try:
        # Get the port from environment variable or use default
        port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
        url = f"http://localhost:{port}/_stcore/health"
        
        # Make a request with a short timeout
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Health check passed - Application is running")
            sys.exit(0)
        else:
            print(f"❌ Health check failed - HTTP {response.status_code}")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed - Connection error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Health check failed - Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_health()
