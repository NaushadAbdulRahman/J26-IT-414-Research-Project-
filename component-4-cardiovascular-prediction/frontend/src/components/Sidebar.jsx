import React from 'react';
import {
  Heart,
  PlusSquare,
  BarChart2,
  HelpCircle,
  Database,
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  onSelectTab,
  cohortStats,
}) {
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
      tag: cohortStats?.total_records ? `${cohortStats.total_records}` : '1,200',
    },
  ];

  return (
    <aside className="app-sidebar" aria-label="Main Navigation Sidebar">
      {/* 1. Sidebar Header / Workspace Brand */}
      <div className="sidebar-header">
        <div className="workspace-brand">
          <div className="workspace-logo-mark">
            <Heart size={18} strokeWidth={2.4} />
          </div>
          <div className="workspace-title-wrap">
            <span className="workspace-eyebrow">Workspace</span>
            <span className="workspace-name">CardioLiver AI</span>
          </div>
        </div>
      </div>

      {/* 2. Primary Navigation Items */}
      <nav className="sidebar-nav" role="tablist">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`sidebar-nav-btn ${item.isPrimary ? 'primary-nav-btn' : ''} ${
                isActive ? 'active' : ''
              }`}
              onClick={() => onSelectTab(item.id)}
              role="tab"
              aria-selected={isActive}
              id={`sidebarNav-${item.id}`}
            >
              <Icon size={17} className="sidebar-nav-icon" strokeWidth={2.2} />
              <span className="sidebar-nav-label">{item.label}</span>
              {item.tag && <span className="sidebar-nav-tag">{item.tag}</span>}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
