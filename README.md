# Federated Adaptive Learning System

A privacy-preserving multi-modal medical diagnostics system using federated learning with real-world dataset integration.

## 🌟 Features

- **Multi-Modal AI**: Combines ECG, voice, handwriting, wearable, and glucose data
- **Federated Learning**: Privacy-preserving collaborative learning across 5 datasets
- **Differential Privacy**: Gradient clipping and noise addition (ε=1.0, δ=1e-5)
- **Explainable AI**: Natural language explanations for every diagnosis
- **Integrated Web Interface**: Patient, Doctor, and Admin portals served by FastAPI

## 📊 Integrated Datasets

| Dataset | Modality | Description |
|---------|----------|-------------|
| PTB-XL | ECG | 21,837 clinical ECG records |
| UCI Parkinson's | Voice | Voice measurements for tremor detection |
| HandPD | Handwriting | Handwriting patterns (synthetic) |
| WESAD | Wearable | Physiological signals from wearables |
| OhioT1DM | Glucose | Continuous glucose monitoring |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Download Datasets (Optional)

```bash
python datasets/download_datasets.py
```

**Note**: Some datasets require manual download:
- **WESAD**: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
- **OhioT1DM**: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

### 3. Start the Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Application

- **Web Interface**: http://localhost:8000
- **Patient Dashboard**: http://localhost:8000/patient
- **Doctor Dashboard**: http://localhost:8000/doctor
- **Admin Dashboard**: http://localhost:8000/admin
- **API Documentation**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/api

## 📱 User Interfaces

### Patient Portal (`/patient`)
- Device pairing (ECG, Voice, Handwriting, Wearable, Glucose)
- Multi-modal diagnosis execution
- Risk scores with XAI explanations
- Privacy consent management

### Doctor Portal (`/doctor`)
- High-risk patient alerts
- Patient insights and multi-modal analysis
- Clinical feedback submission
- No access to raw patient data

### Admin Portal (`/admin`)
- System statistics
- Federated learning monitoring
- Dataset status tracking
- Privacy budget tracking

## 🔬 Running a Diagnosis

1. Navigate to **Patient Portal** (http://localhost:8000/patient)
2. Ensure devices are paired (pre-paired in demo)
3. Click **"Run New Diagnosis"**
4. View multi-modal risk scores:
   - CVD Risk
   - Parkinson's Risk
   - Diabetes Risk
5. Read XAI explanations and recommendations

## 🔐 Privacy Features

- **Differential Privacy**: ε=1.0, δ=1e-5
- **Gradient Clipping**: Max norm = 1.0
- **Secure Aggregation**: Encrypted model updates
- **No Raw Data Sharing**: Only model gradients leave devices

## 🧪 Testing ML Models

```bash
# Test individual encoders
cd backend
python app/ml/encoders/ecg_encoder.py
python app/ml/encoders/voice_encoder.py
python app/ml/encoders/handwriting_encoder.py
python app/ml/encoders/wearable_encoder.py
python app/ml/encoders/glucose_encoder.py

# Test fusion
python app/ml/fusion/multimodal_fusion.py

# Test explainability
python app/ml/xai/explainer.py

# Test federated learning
python app/ml/federated/server.py

# Test differential privacy
python app/ml/privacy/differential_privacy.py
```

## 📡 API Endpoints

### Web Pages
- `GET /` - Landing page
- `GET /patient` - Patient dashboard
- `GET /doctor` - Doctor dashboard
- `GET /admin` - Admin dashboard

### API Endpoints
- `GET /api` - API information
- `GET /health` - Health check
- `POST /api/patients/register` - Register new patient
- `POST /api/devices/diagnose/{patient_id}` - Run diagnosis
- `GET /api/doctors/dashboard` - Get doctor dashboard data
- `POST /api/federated/rounds/start` - Start FL round

Full API documentation: http://localhost:8000/docs

## 🏗️ Architecture

```
Backend (FastAPI)
├── Templates (Jinja2)          # HTML pages
├── Static Files                # CSS
├── API Routes                  # REST endpoints
├── ML Models (PyTorch)
│   ├── Encoders (5 modalities)
│   ├── Multi-Modal Fusion
│   ├── Federated Learning
│   └── Explainability
└── Database (SQLite)
```

## 🗂️ Project Structure

```
Federated-Adaptive-Learning-System-/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app with HTML routes
│   │   ├── routers/                # API endpoints
│   │   ├── services/               # Business logic
│   │   └── ml/                     # ML components
│   ├── templates/                  # HTML templates (Jinja2)
│   │   ├── index.html
│   │   ├── patient_dashboard.html
│   │   ├── doctor_dashboard.html
│   │   └── admin_dashboard.html
│   ├── static/                     # Static files
│   │   └── css/
│   │       └── global.css
│   ├── datasets/                   # Dataset loaders
│   └── requirements.txt
├── database/
│   ├── schema.sql
│   └── seed.sql
└── README.md
```

## 🎯 Key Technologies

- **Backend**: FastAPI, Jinja2 Templates, PyTorch (CPU), SQLite
- **Frontend**: HTML, JavaScript, CSS (served by FastAPI)
- **ML**: CNN, LSTM, Transformer, Attention Mechanism
- **Privacy**: Differential Privacy, Federated Learning
- **Datasets**: PTB-XL, UCI, HandPD, WESAD, OhioT1DM

## 📝 Development

### Adding New Pages

1. Create HTML template in `backend/templates/`
2. Add route in `backend/app/main.py`:

```python
@app.get("/your-page", response_class=HTMLResponse)
async def your_page(request: Request):
    return templates.TemplateResponse("your_template.html", {"request": request})
```

### Adding New API Endpoints

1. Create router in `backend/app/routers/`
2. Include in `main.py`:

```python
app.include_router(your_router, prefix="/api/your-prefix", tags=["Your Tag"])
```

## 📧 Support

For questions or issues, please refer to the implementation plan and walkthrough in the `brain/` directory.

## 🎉 Summary

This is a **complete, integrated federated learning system** with:
- **30+ Python modules** implementing state-of-the-art ML
- **4 web pages** served directly by FastAPI
- **20+ API endpoints** for full system control
- **5 real-world datasets** integrated
- **Privacy-preserving** federated learning
- **Explainable AI** for clinical trust

**No separate frontend server needed** - everything runs from one FastAPI application!

## 🐳 Docker Support

You can run the application using Docker to ensure a consistent environment.

### Prerequisites
- Docker
- Docker Compose

### 1. Build and Start
```bash
docker-compose up --build
```

### 2. Access
Same as local setup: http://localhost:8000
