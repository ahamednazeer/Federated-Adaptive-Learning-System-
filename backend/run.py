#!/usr/bin/env python3
"""
Run script for Federated Adaptive Learning System
Simple script to start the FastAPI server
"""

import uvicorn
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🏥 Starting Federated Adaptive Learning System...")
    print("=" * 60)
    print("Server will be available at:")
    print("  - Landing Page: http://localhost:8000")
    print("  - Patient Portal: http://localhost:8000/patient")
    print("  - Doctor Portal: http://localhost:8000/doctor")
    print("  - Admin Portal: http://localhost:8000/admin")
    print("  - API Docs: http://localhost:8000/docs")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
