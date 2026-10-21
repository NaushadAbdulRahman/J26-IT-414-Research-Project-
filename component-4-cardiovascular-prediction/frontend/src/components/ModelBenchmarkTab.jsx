import React, { useState, useEffect } from 'react';
import { Activity, BarChart3 } from 'lucide-react';

export default function ModelBenchmarkTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/model_comparison')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch benchmark metrics');
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="card loading-card" style={{ padding: 40, textAlign: 'center' }}>
        <Activity size={24} className="spin-icon" style={{ margin: '0 auto 12px' }} />
        <p style={{ color: 'var(--text-muted)' }}>
          Evaluating AI models, calculating incremental discrimination gain (ΔAUC), and building decision curves...
        </p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card error-alert">
        <p>Error loading benchmark: {error}</p>
      </div>
    );
  }

  const { incremental, metrics, dca, calibration } = data;
  const algos = ['Logistic Regression', 'Random Forest', 'XGBoost'];

  return (
    <div className="full-tab-content">
      {/* 1. Header & Summary Stats */}
      <div className="card intro-card">
        <div className="intro-header">
          <span className="step-badge green-badge">INFOGRAPHIC SECTION 03 & SPECIAL TECHNIQUES</span>
          <h2>Model Development, Benchmark & Incremental Value Analysis</h2>
          <p>
            Evaluating whether adding liver fibrosis severity (Model B) provides significant incremental prognostic
            discrimination over traditional risk factor scores (Model A) across <strong>Logistic Regression</strong>,{' '}
            <strong>Random Forest</strong>, and <strong>XGBoost</strong> algorithms.
          </p>
        </div>

        {/* 4 Executive KPI Metric Tiles */}
        <div className="kpi-banner-grid">
          <div className="kpi-metric-tile">
            <span className="kpi-caption">INCREMENTAL DISCRIMINATION GAIN</span>
            <div className="kpi-figure mono">+{incremental.delta_auc.toFixed(3)} ΔAUC</div>
            <span className="kpi-footnote">Consistent elevation in stratified test set</span>
          </div>
          <div className="kpi-metric-tile" style={{ borderTopColor: '#0284c7' }}>
            <span className="kpi-caption">NET RECLASSIFICATION IMPROVEMENT (NRI)</span>
            <div className="kpi-figure mono" style={{ color: '#0284c7' }}>+{incremental.nri.toFixed(3)}</div>
            <span className="kpi-footnote">Correct reclassification of event/non-event cases</span>
          </div>
          <div className="kpi-metric-tile" style={{ borderTopColor: '#7c3aed' }}>
            <span className="kpi-caption">INTEGRATED DISCRIMINATION (IDI)</span>
            <div className="kpi-figure mono" style={{ color: '#7c3aed' }}>+{incremental.idi.toFixed(3)}</div>
            <span className="kpi-footnote">Average difference in predicted probabilities</span>
          </div>
          <div className="kpi-metric-tile" style={{ borderTopColor: '#0f2b5c' }}>
            <span className="kpi-caption">TOP PERFORMING ALGORITHM</span>
            <div className="kpi-figure mono" style={{ color: '#0f2b5c', fontSize: '20px' }}>{incremental.enhanced_best}</div>
            <span className="kpi-footnote">Selected by optimal ROC-AUC & Brier score</span>
          </div>
        </div>
      </div>

      {/* 2. Algorithm Evaluation Matrix Table */}
      <div className="card">
        <div className="card-header-bar">
          <div className="header-icon-box blue">
            <BarChart3 size={18} strokeWidth={2.3} />
          </div>
          <div>
            <span className="step-badge blue-badge">PERFORMANCE MATRIX</span>
            <h2 className="card-title">Comprehensive Algorithm Evaluation Matrix (Stratified Test Cohort, N=240)</h2>
            <span className="section-sub">Model A (Baseline Traditional CVD) vs Model B (Fibrosis-Enhanced)</span>
          </div>
        </div>

        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table className="clinical-table">
            <thead>
              <tr>
                <th>Model Specification</th>
                <th>Algorithm</th>
                <th>AUC-ROC</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
                <th>F1-Score</th>
                <th>Accuracy</th>
                <th>Brier Score</th>
              </tr>
            </thead>
            <tbody>
              {algos.map((algo) => {
                const base = metrics.baseline[algo];
                const enh = metrics.enhanced[algo];
                const isBest = algo === incremental.enhanced_best;

                return (
                  <React.Fragment key={algo}>
                    <tr>
                      <td><span className="tag-baseline">Model A (Baseline)</span></td>
                      <td style={{ fontWeight: 600 }}>{algo}</td>
                      <td className="mono">{base?.auc?.toFixed(3)}</td>
                      <td className="mono">{base?.sensitivity?.toFixed(3)}</td>
                      <td className="mono">{base?.specificity?.toFixed(3)}</td>
                      <td className="mono">{base?.f1?.toFixed(3)}</td>
                      <td className="mono">{base?.accuracy?.toFixed(3)}</td>
                      <td className="mono">{base?.brier?.toFixed(3)}</td>
                    </tr>
                    <tr className={isBest ? 'highlight-row' : ''}>
                      <td><span className="tag-enhanced">Model B (Fibrosis-Enhanced)</span></td>
                      <td style={{ fontWeight: 700, color: '#0f766e' }}>
                        {algo} {isBest && '★ (Best Model)'}
                      </td>
                      <td className="mono" style={{ fontWeight: 800, color: '#0f766e' }}>{enh?.auc?.toFixed(3)}</td>
                      <td className="mono">{enh?.sensitivity?.toFixed(3)}</td>
                      <td className="mono">{enh?.specificity?.toFixed(3)}</td>
                      <td className="mono">{enh?.f1?.toFixed(3)}</td>
                      <td className="mono">{enh?.accuracy?.toFixed(3)}</td>
                      <td className="mono" style={{ fontWeight: 800, color: '#0f766e' }}>{enh?.brier?.toFixed(3)}</td>
                    </tr>
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

