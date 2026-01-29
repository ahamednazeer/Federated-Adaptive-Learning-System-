-- Users table (Patient, Doctor, Admin)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('patient', 'doctor', 'admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extended patient profile
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    age INTEGER,
    medical_history TEXT,
    consent_privacy BOOLEAN DEFAULT 0,
    consent_federated_learning BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Edge devices mapped to datasets
CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    device_type TEXT NOT NULL CHECK(device_type IN ('ecg', 'voice', 'handwriting', 'wearable', 'glucose')),
    dataset_name TEXT NOT NULL CHECK(dataset_name IN ('PTB-XL', 'UCI', 'HandPD', 'WESAD', 'OhioT1DM')),
    status TEXT DEFAULT 'paired' CHECK(status IN ('paired', 'active', 'offline')),
    last_sync TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Data collection sessions
CREATE TABLE IF NOT EXISTS data_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER NOT NULL,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    samples_collected INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'failed')),
    FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE
);

-- Multi-modal diagnosis results
CREATE TABLE IF NOT EXISTS diagnoses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    cvd_risk REAL,
    parkinsons_risk REAL,
    diabetes_risk REAL,
    fusion_embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- XAI explanations
CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id INTEGER NOT NULL,
    modality TEXT NOT NULL CHECK(modality IN ('ecg', 'voice', 'handwriting', 'wearable', 'glucose', 'fusion')),
    attention_weights TEXT,
    saliency_map BLOB,
    natural_language TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE
);

-- Federated learning rounds
CREATE TABLE IF NOT EXISTS federated_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'active', 'completed', 'failed')),
    num_clients INTEGER DEFAULT 0,
    global_loss REAL,
    global_accuracy REAL,
    privacy_epsilon REAL,
    privacy_delta REAL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Encrypted model updates from edge devices
CREATE TABLE IF NOT EXISTS model_updates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id INTEGER NOT NULL,
    client_name TEXT NOT NULL,
    dataset_partition TEXT NOT NULL,
    gradient_blob BLOB,
    local_loss REAL,
    local_accuracy REAL,
    dp_noise_applied BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (round_id) REFERENCES federated_rounds(id) ON DELETE CASCADE
);

-- Doctor feedback for continual learning
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    diagnosis_id INTEGER NOT NULL,
    doctor_id INTEGER NOT NULL,
    confirmed BOOLEAN,
    corrected_cvd_risk REAL,
    corrected_parkinsons_risk REAL,
    corrected_diabetes_risk REAL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE,
    FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE
);

-- System logs
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL CHECK(level IN ('INFO', 'WARNING', 'ERROR')),
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients(user_id);
CREATE INDEX IF NOT EXISTS idx_devices_patient_id ON devices(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_patient_id ON diagnoses(patient_id);
CREATE INDEX IF NOT EXISTS idx_explanations_diagnosis_id ON explanations(diagnosis_id);
CREATE INDEX IF NOT EXISTS idx_federated_rounds_status ON federated_rounds(status);
CREATE INDEX IF NOT EXISTS idx_model_updates_round_id ON model_updates(round_id);
CREATE INDEX IF NOT EXISTS idx_feedback_diagnosis_id ON feedback(diagnosis_id);
