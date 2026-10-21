import React, { useState, useEffect } from 'react';
import { UploadCloud, Download, CheckCircle, AlertTriangle, Database } from 'lucide-react';

export default function CohortDataTab({ onUploadSuccess }) {
  const [status, setStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  const fetchStatus = () => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch((err) => console.error('Error fetching cohort status:', err));
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Please select a valid .csv file.');
      return;
    }

    setUploading(true);
    setUploadMessage(null);
    setUploadError(null);

    const formData = new FormData();
    formData.append('dataset', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.detail?.error || json.detail || json.error || 'Upload failed');
      }

      setUploadMessage(json.message);
      fetchStatus();
      if (onUploadSuccess) onUploadSuccess();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="full-tab-content">
      {/* 1. Cohort Lead Card */}
      <div className="card cohort-lead-card">
        <div className="cohort-lead-left">
          <span className="step-badge teal-badge">DATASET SPECIFICATION · MASLD CLINICAL REGISTRY</span>
          <h2>MASLD Patient Cohort Overview</h2>
          <p>
            Cardiovascular risk and fibrosis surveillance cohort comprising <strong>{status?.total_records?.toLocaleString() || '1,200'} patients</strong> with 22 clinical variables, an observed CVD event rate of <strong>{status?.cvd_event_rate || 22.2}%</strong>, and <strong>{status?.total_missing_values ?? 24} missing observations</strong> robustly preprocessed via median/mode imputation.
          </p>

          <div className="download-action-row">
            <a
              href="http://127.0.0.1:8000/api/download_template"
              download="masld_patient_cohort_template.csv"
              className="btn-download"
            >
              <Download size={14} />
              <span>Download Dataset Template (.CSV)</span>
            </a>
            <span className="action-note">Pre-formatted schema with sample distributions</span>
          </div>
        </div>

        {/* Upload Widget */}
        <div className="upload-widget">
          <div className="card-header-bar" style={{ marginBottom: 12, paddingBottom: 10 }}>
            <div className="header-icon-box lilac">
              <UploadCloud size={18} strokeWidth={2.3} />
            </div>
            <div>
              <span className="step-badge purple-badge">REAL-TIME INGESTION</span>
              <h3>Upload New Patient Cohort</h3>
            </div>
          </div>

          <p style={{ fontSize: '11.5px', color: 'var(--text-muted)', marginBottom: 12 }}>
            Upload external CSV data to immediately validate variables, retrain all ML models, and update metrics.
          </p>

          {uploadMessage && (
            <div className="success-alert" style={{ marginBottom: 12, padding: '8px 12px', fontSize: '12px' }}>
              <CheckCircle size={15} />
              <span>{uploadMessage}</span>
            </div>
          )}

          {uploadError && (
            <div className="error-alert" style={{ marginBottom: 12, padding: '8px 12px', fontSize: '12px' }}>
              <AlertTriangle size={15} />
              <span>{uploadError}</span>
            </div>
          )}

          <label
            htmlFor="csvUploadInput"
            style={{
              display: 'block',
              padding: '18px 12px',
              border: '2px dashed #cbd5e1',
              borderRadius: '8px',
              textAlign: 'center',
              cursor: 'pointer',
              background: '#ffffff',
            }}
          >
            <UploadCloud size={28} style={{ color: 'var(--primary-teal)', margin: '0 auto 6px' }} />
            <div style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-main)' }}>
              Click to browse or drop CSV here
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: 2 }}>
              Requires at least 30 rows with cvd_event
            </div>
            <input
              id="csvUploadInput"
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              disabled={uploading}
            />
            {uploading && (
              <div style={{ marginTop: 8, fontSize: '11.5px', fontWeight: 600, color: 'var(--primary-teal)' }}>
                Retraining models with new cohort...
              </div>
            )}
          </label>
        </div>
      </div>
    </div>
  );
}
