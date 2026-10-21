import React from 'react';
import { ClipboardList, BarChart3, HelpCircle, UploadCloud } from 'lucide-react';

export default function ProjectBanner({ activeTab, onSelectTab }) {
  const tabs = [
    {
      id: 'decision-support',
      title: 'Patient Decision Support',
      subtitle: 'Individual CVD Risk & Guidance',
      tag: 'Live',
      isLive: true,
      icon: ClipboardList,
    },
    {
      id: 'model-comparison',
      title: 'Model Benchmark',
      subtitle: 'Model A vs. Model B (Incremental)',
      tag: 'ΔAUC',
      icon: BarChart3,
    },
    {
      id: 'explainable-ai',
      title: 'Explainable AI (SHAP)',
      subtitle: 'Global Feature Synergy & Attribution',
      tag: 'SHAP',
      icon: HelpCircle,
    },
    {
      id: 'cohort-management',
      title: 'Cohort Data & Upload',
      subtitle: 'Patient Registry & CSV Ingestion',
      tag: '1,200',
      icon: UploadCloud,
    },
  ];

  return (
    <section className="project-banner">
      <div className="banner-inner">
        <div className="banner-title-row">
          <div className="banner-badge">CLINICAL DECISION SUPPORT FRAMEWORK</div>
          <h1 className="project-title">
            AI-Assisted Cardiovascular Risk Prediction and Explainable Clinical Decision Support in MASLD Patients
          </h1>
        </div>

        <nav className="clinical-nav-grid" role="tablist" aria-label="Clinical Decision Framework Navigation">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                className={`nav-tab-btn ${isActive ? 'active' : ''}`}
                onClick={() => onSelectTab(tab.id)}
                role="tab"
                aria-selected={isActive}
                id={`tabBtn-${tab.id}`}
              >
                <div className="nav-tab-icon">
                  <Icon size={20} strokeWidth={2.2} />
                </div>
                <div className="nav-tab-content">
                  <div className="nav-tab-title-row">
                    <span className="nav-tab-title">{tab.title}</span>
                    <span className={`nav-tab-tag ${tab.isLive ? 'live-tag' : ''}`}>{tab.tag}</span>
                  </div>
                  <span className="nav-tab-subtitle">{tab.subtitle}</span>
                </div>
              </button>
            );
          })}
        </nav>
      </div>
    </section>
  );
}
