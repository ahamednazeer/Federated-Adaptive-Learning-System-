-- Sample users
INSERT INTO users (email, password_hash, role) VALUES
('patient1@example.com', 'hashed_password_1', 'patient'),
('patient2@example.com', 'hashed_password_2', 'patient'),
('doctor1@example.com', 'hashed_password_3', 'doctor'),
('admin1@example.com', 'hashed_password_4', 'admin');

-- Sample patients
INSERT INTO patients (user_id, name, age, medical_history, consent_privacy, consent_federated_learning) VALUES
(1, 'John Doe', 65, 'Hypertension, Family history of CVD', 1, 1),
(2, 'Jane Smith', 58, 'Type 2 Diabetes, Obesity', 1, 1);

-- Sample devices (mapped to datasets)
INSERT INTO devices (patient_id, device_type, dataset_name, status, last_sync) VALUES
(1, 'ecg', 'PTB-XL', 'active', CURRENT_TIMESTAMP),
(1, 'voice', 'UCI', 'active', CURRENT_TIMESTAMP),
(1, 'handwriting', 'HandPD', 'active', CURRENT_TIMESTAMP),
(2, 'wearable', 'WESAD', 'active', CURRENT_TIMESTAMP),
(2, 'glucose', 'OhioT1DM', 'active', CURRENT_TIMESTAMP);
