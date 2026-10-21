import React, { useState, useEffect } from 'react';
import { Sparkles, Activity } from 'lucide-react';

export default function ExplainableAITab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/model_comparison')
      .then((r) => r.json())
      .then((compData) => {
        setData(compData);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error loading XAI data:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="card loading-card" style={{ padding: 40, textAlign: 'center' }}>
        <Activity size={24} className="spin-icon" style={{ margin: '0 auto 12px' }} />
        <p style={{ color: 'var(--text-muted)' }}>
          Loading Explainable AI (SHAP) global feature importance values...
        </p>
      </div>
    );
  }

  const globalShap = data?.global_shap || [];

  return (
    <div className="full-tab-content">
      {/* 1. Global SHAP Feature Importance */}
      <div className="card">
        <div className="card-header-bar">
          <div className="header-icon-box blue">
            <Sparkles size={18} strokeWidth={2.3} />
          </div>
          <div>
            <span className="step-badge green-badge">BOX 5 · GLOBAL EXPLAINABILITY (SHAP TREEXPLAINER)</span>
            <h2 className="card-title">Population-Level Feature Importance & Fibrosis Impact</h2>
            <span className="section-sub">
              Mean absolute SHAP value across the entire cohort (N=1,200) quantifying feature impact on 10-year CVD risk
            </span>
          </div>
        </div>

        <div className="global-shap-list" style={{ marginTop: 16 }}>
          {globalShap.slice(0, 10).map((feat, idx) => {
            const maxImp = globalShap[0]?.importance || 1.0;
            const barWidth = Math.min((feat.importance / maxImp) * 100, 100);

            return (
              <div key={idx} className="global-row">
                <div className="global-label">
                  #{idx + 1} {feat.feature}
                  {feat.is_fibrosis && (
                    <span
                      style={{
                        marginLeft: 6,
                        fontSize: '10px',
                        fontWeight: 700,
                        color: '#0f766e',
                        background: '#ccfbf1',
                        padding: '1px 6px',
                        borderRadius: '4px',
                      }}
                    >
                      FIBROSIS
                    </span>
                  )}
                </div>
                <div className="global-track">
                  <div
                    className="global-fill"
                    style={{
                      width: `${barWidth}%`,
                      background: feat.is_fibrosis
                        ? 'linear-gradient(90deg, #f59e0b, #d97706)'
                        : 'linear-gradient(90deg, #14b8a6, #0f766e)',
                    }}
                  ></div>
                </div>
                <div className="global-val mono">{feat.importance.toFixed(3)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
