import React from 'react';
import { Activity, Heart, Printer, ExternalLink } from 'lucide-react';

export default function Header({ cohortStats }) {
  const handlePrint = () => {
    window.print();
  };

  return (
    <header className="app-navbar">
      <div className="nav-container">
        <div className="nav-brand">
          <div className="brand-logo">
            <Heart size={20} strokeWidth={2.3} />
          </div>
          <div className="brand-info">
            <div className="brand-name">
              CardioLiver<span className="brand-ai">AI</span>
            </div>
            <div className="brand-institution">MASLD Clinical Research Platform</div>
          </div>
        </div>

        <div className="nav-status-group">
          <div className="status-chip active">
            <span className="status-pulse"></span>
            <span>
              MASLD Cohort: <strong>{cohortStats?.total_records?.toLocaleString() || '1,200'} Patients</strong>
            </span>
          </div>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="status-chip highlight"
            style={{ textDecoration: 'none', cursor: 'pointer' }}
            title="Open interactive FastAPI Swagger documentation"
          >
            <Activity size={14} strokeWidth={2.5} />
            <span>FastAPI Docs (Swagger)</span>
            <ExternalLink size={12} style={{ marginLeft: 3, opacity: 0.8 }} />
          </a>

          <button type="button" className="btn-print" onClick={handlePrint} title="Print clinical report">
            <Printer size={15} strokeWidth={2} />
            Print Report
          </button>
        </div>
      </div>
    </header>
  );
}
