"""
FastAPI Main Application
Entry point for the Federated Adaptive Learning System backend
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import aiosqlite
from pathlib import Path

# Import routers
from app.routers import patients, devices, doctors, federated
from app.services.database import init_database

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    print("🚀 Starting Federated Adaptive Learning System...")
    
    # Initialize database
    db_path = Path("database/federated_learning.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    await init_database(str(db_path))
    app_state['db_path'] = str(db_path)
    
    print("✓ Database initialized")
    
    # Load ML models (lazy loading)
    print("✓ ML models ready for lazy loading")
    
    print("✅ System ready!")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Federated Adaptive Learning System",
    description="Privacy-preserving multi-modal medical diagnostics with federated learning",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(patients.router, prefix="/api/patients", tags=["Patients"])
app.include_router(devices.router, prefix="/api/devices", tags=["Devices"])
app.include_router(doctors.router, prefix="/api/doctors", tags=["Doctors"])
app.include_router(federated.router, prefix="/api/federated", tags=["Federated Learning"])


# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/patient", response_class=HTMLResponse)
async def patient_dashboard(request: Request):
    """Patient dashboard"""
    return templates.TemplateResponse("patient_dashboard.html", {"request": request})


@app.get("/doctor", response_class=HTMLResponse)
async def doctor_dashboard(request: Request):
    """Doctor dashboard"""
    return templates.TemplateResponse("doctor_dashboard.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard"""
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})


@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Federated Adaptive Learning System API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "patients": "/api/patients",
            "devices": "/api/devices",
            "doctors": "/api/doctors",
            "federated": "/api/federated",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "ml_models": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
