#!/usr/bin/env python3
"""
Simple script to run the UMBC Academic Risk Predictor API server
"""

import uvicorn
import os
import sys

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == "__main__":
    print("Starting UMBC Academic Risk Predictor API...")
    print("Server will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Frontend should connect to: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
