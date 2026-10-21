import React, { useState } from 'react';
import DecisionSupportTab from './components/DecisionSupportTab';
import ModelBenchmarkTab from './components/ModelBenchmarkTab';
import ExplainableAITab from './components/ExplainableAITab';
import CohortDataTab from './components/CohortDataTab';
import {
  Heart,
  PlusSquare,
  BarChart2,
  HelpCircle,
  Database,
  Printer,
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('decision-support');

  const navItems = [
    {
      id: 'decision-support',
      label: 'New Assessment',
      icon: PlusSquare,
      isPrimary: true,
      tag: 'Live',
    },
    {
      id: 'model-comparison',
      label: 'Model Benchmarks',
      icon: BarChart2,
      tag: 'ΔAUC',
    },
    {
      id: 'explainable-ai',
      label: 'Explainable AI (SHAP)',
      icon: HelpCircle,
      tag: 'SHAP',
    },
    {
      id: 'cohort-management',
      label: 'Cohort Registry',
      icon: Database,
    },
  ];

  const handlePrint = () => {
    window.print();
  };

  // Title and subtitle dynamically matching active tab
  const getTabHeader = () => {
    switch (activeTab) {
      case 'decision-support':
        return {
          title: 'Patient CVD Risk Assessment',
          subtitle: 'Provide patient clinical parameters or select a reference case to generate 10-year CVD risk.',
        };
      case 'model-comparison':
        return {
          title: 'Predictive Model Evaluation & Incremental Value',
          subtitle: 'Evaluating incremental prognostic gain (ΔAUC, NRI, IDI) of liver fibrosis across algorithms.',
        };
      case 'explainable-ai':
        return {
          title: 'Explainable AI & Biological Synergy (SHAP)',
          subtitle: 'Quantifying population-level feature importance and non-linear metabolic-fibrotic interactions.',
        };
      case 'cohort-management':
        return {
          title: 'MASLD Patient Cohort & Real-Time Ingestion',
          subtitle: 'Sri Lanka MASLD clinical surveillance registry with 1,200 patients and 22 clinical variables.',
        };
      default:
        return {
          title: 'CardioLiver-AI Decision Support',
          subtitle: 'AI-assisted cardiovascular risk stratification in Metabolic Dysfunction-Associated Steatotic Liver Disease.',
        };
    }
  };

  const headerMeta = getTabHeader();

  return (
    <div className="app-single-canvas-layout">
      {/* 1. Clean Top Header with Attached Navigation (No Sidebar) */}
      <header className="app-top-header" role="banner">
        <div className="header-container">
          {/* Brand Mark */}
          <div className="header-brand">
            <div className="brand-logo-mark">
              <Heart size={18} strokeWidth={2.4} />
            </div>
            <div className="brand-text-wrap">
              <span className="brand-title">CardioLiver AI</span>
              <span className="brand-subtitle">Clinical Platform</span>
            </div>
          </div>

          {/* Attached Navigation Tabs in Header */}
          <nav className="header-nav-tabs" role="tablist" aria-label="Clinical workspace navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`header-nav-btn ${item.isPrimary ? 'primary-pill' : ''} ${
                    isActive ? 'active' : ''
                  }`}
                  onClick={() => setActiveTab(item.id)}
                  role="tab"
                  aria-selected={isActive}
                  id={`headerNav-${item.id}`}
                >
                  <Icon size={16} strokeWidth={2.2} />
                  <span>{item.label}</span>
                  {item.tag && <span className="header-nav-tag">{item.tag}</span>}
                </button>
              );
            })}
          </nav>

          {/* Right Action: Print Report */}
          <div className="header-actions">
            <button
              type="button"
              className="header-btn-print"
              onClick={handlePrint}
              title="Print clinical assessment report"
            >
              <Printer size={14} strokeWidth={2.2} />
              <span>Print Report</span>
            </button>
          </div>
        </div>
      </header>

      {/* 2. Main Content Canvas */}
      <main className="main-content-canvas" role="main">
        {/* Tab Specific Workspace Content */}
        <div className="canvas-body-container">
          {activeTab === 'decision-support' && <DecisionSupportTab />}
          {activeTab === 'model-comparison' && <ModelBenchmarkTab />}
          {activeTab === 'explainable-ai' && <ExplainableAITab />}
          {activeTab === 'cohort-management' && <CohortDataTab />}
        </div>
      </main>
    </div>
  );
}
