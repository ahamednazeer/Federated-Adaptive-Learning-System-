#!/bin/bash

# Federated Adaptive Learning System - Quick Start Script

echo "🏥 Federated Adaptive Learning System - Quick Start"
echo "=================================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo ""
echo "Initializing database..."
python3 -c "import asyncio; from app.services.database import init_database; asyncio.run(init_database('../database/federated_learning.db'))"

# Download datasets (optional)
echo ""
echo "Dataset download (optional)..."
echo "Run: python datasets/download_datasets.py"
echo "Note: Some datasets require manual download"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit:"
echo "  Landing Page: http://localhost:8000"
echo "  Patient Portal: http://localhost:8000/patient"
echo "  Doctor Portal: http://localhost:8000/doctor"
echo "  Admin Portal: http://localhost:8000/admin"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "No separate frontend server needed!"
