import React, { useState } from 'react';
import {
  User,
  Activity,
  AlertTriangle,
  ShieldCheck,
  Layers,
  Sparkles,
  Sliders,
  UserCheck,
  FolderOpen,
  RotateCcw,
} from 'lucide-react';

const PRESETS = {
  F3: {
    age: 56, sex: 'Female', bmi: 31.2, diabetes: 'Yes', hypertension: 'Yes', smoker: 'No',
    systolic_bp: 146, diastolic_bp: 92, fasting_glucose: 128, total_cholesterol: 218,
    ldl: 138, hdl: 42, triglycerides: 190, ast: 54, alt: 70, platelet_count: 175,
    fibrosis_stage: 3, slcvd_score: 18.2, who_cvd_risk: 15.0,
  },
  F2: {
    age: 52, sex: 'Male', bmi: 28.4, diabetes: 'No', hypertension: 'Yes', smoker: 'Yes',
    systolic_bp: 138, diastolic_bp: 86, fasting_glucose: 104, total_cholesterol: 205,
    ldl: 130, hdl: 46, triglycerides: 165, ast: 38, alt: 52, platelet_count: 210,
    fibrosis_stage: 2, slcvd_score: 12.5, who_cvd_risk: 10.0,
  },
  F0: {
    age: 44, sex: 'Female', bmi: 24.8, diabetes: 'No', hypertension: 'No', smoker: 'No',
    systolic_bp: 118, diastolic_bp: 76, fasting_glucose: 92, total_cholesterol: 175,
    ldl: 102, hdl: 58, triglycerides: 110, ast: 24, alt: 28, platelet_count: 275,
    fibrosis_stage: 0, slcvd_score: 4.8, who_cvd_risk: 5.0,
  },
};

const BLANK_PATIENT = {
  age: '',
  sex: 'Female',
  bmi: '',
  diabetes: 'No',
  hypertension: 'No',
  smoker: 'No',
  systolic_bp: '',
  diastolic_bp: '',
  fasting_glucose: '',
  total_cholesterol: '',
  ldl: '',
  hdl: '',
  triglycerides: '',
  platelet_count: '',
  ast: '',
  alt: '',
  fibrosis_stage: '0',
  slcvd_score: '10.0',
  who_cvd_risk: '8.0',
};

const STANDARD_BASELINE = {
  age: 52,
  sex: 'Male',
  bmi: 27.2,
  diabetes: 'No',
  hypertension: 'No',
  smoker: 'No',
  systolic_bp: 126,
  diastolic_bp: 82,
  fasting_glucose: 98,
  total_cholesterol: 195,
  ldl: 118,
  hdl: 52,
  triglycerides: 145,
  platelet_count: 240,
  ast: 28,
  alt: 32,
  fibrosis_stage: '0',
  slcvd_score: '8.5',
  who_cvd_risk: '7.0',
};

function getFriendlyFactorName(item) {
  const f = (item.feature || '').toLowerCase();
  if (f === 'age') return 'Age';
  if (f.includes('diabetes')) return 'Diabetes';
  if (f.includes('fibrosis') || f === 'fibrosis_stage') return 'Severe Fibrosis';
  if (f.includes('hypertension')) return 'Hypertension';
  if (f === 'bmi') return 'BMI';
  if (f.includes('systolic')) return 'Systolic BP';
  if (f.includes('diastolic')) return 'Diastolic BP';
  if (f.includes('glucose')) return 'Fasting Glucose';
  if (f.includes('fib_4') || f.includes('fib4')) return 'FIB-4 Index';
  if (f === 'ast') return 'AST';
  if (f === 'alt') return 'ALT';
  if (f.includes('cholesterol')) return 'Total Cholesterol';
  if (f === 'ldl') return 'LDL-C';
  if (f === 'hdl') return 'HDL-C';
  if (f.includes('triglycerides')) return 'Triglycerides';
  if (f.includes('platelet')) return 'Platelet Count';
  if (f.includes('smoker')) return 'Smoker';
  if (f.includes('slcvd')) return 'SLCVD Baseline';
  if (f.includes('who')) return 'WHO CVD Risk';
  return item.feature.replace(/_/g, ' ');
}

export default function DecisionSupportTab() {
  // Input mode: 'user' is the primary default mode, 'preset' provides preloaded reference cases
  const [inputMode, setInputMode] = useState('user');
  const [userFormData, setUserFormData] = useState(BLANK_PATIENT);
  const [presetFormData, setPresetFormData] = useState(PRESETS.F3);
  const [activePreset, setActivePreset] = useState('F3');

  const [userPrediction, setUserPrediction] = useState(null);
  const [presetPrediction, setPresetPrediction] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('Model B (Fibrosis-Enhanced Logistic Regression)');

  const currentFormData = inputMode === 'user' ? userFormData : presetFormData;
  const currentPrediction = inputMode === 'user' ? userPrediction : presetPrediction;

  const isFibrosisModel =
    selectedModel.toLowerCase().includes('enhanced') || selectedModel.toLowerCase().includes('model b');

  const calculateFIB4 = (dataToUse = currentFormData) => {
    const age = parseFloat(dataToUse.age);
    const ast = parseFloat(dataToUse.ast);
    const alt = parseFloat(dataToUse.alt);
    const plt = parseFloat(dataToUse.platelet_count);

    if (isNaN(age) || isNaN(ast) || !alt || !plt) {
      return '—';
    }
    return ((age * ast) / (plt * Math.sqrt(Math.max(alt, 1)))).toFixed(2);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const updatedVal = ['sex', 'diabetes', 'hypertension', 'smoker'].includes(name)
      ? value
      : value === ''
      ? ''
      : parseFloat(value) || value;

    if (inputMode === 'user') {
      setUserFormData((prev) => ({ ...prev, [name]: updatedVal }));
    } else {
      setPresetFormData((prev) => ({ ...prev, [name]: updatedVal }));
      setActivePreset(null);
    }
  };

  const handleSwitchMode = (newMode) => {
    setInputMode(newMode);
    setError(null);
    if (newMode === 'preset') {
      if (!presetPrediction) {
        handleCalculateRisk(selectedModel, presetFormData, 'preset');
      }
    }
  };

  const loadPreset = (key) => {
    const selectedCase = PRESETS[key];
    setPresetFormData(selectedCase);
    setActivePreset(key);
    handleCalculateRisk(selectedModel, selectedCase, 'preset');
  };

  const handleFillBaseline = () => {
    setUserFormData(STANDARD_BASELINE);
    setError(null);
  };

  const handleClearFields = () => {
    setUserFormData(BLANK_PATIENT);
    setUserPrediction(null);
    setError(null);
  };

  const handleCalculateRisk = async (modelOverride, dataOverride, modeOverride) => {
    const targetMode = modeOverride !== undefined ? modeOverride : inputMode;
    const dataToUse = dataOverride !== undefined ? dataOverride : (targetMode === 'user' ? userFormData : presetFormData);
    const modelToUse = modelOverride !== undefined ? modelOverride : selectedModel;
    const isFibrosis = modelToUse.toLowerCase().includes('enhanced') || modelToUse.toLowerCase().includes('model b');

    if (!dataToUse.age || !dataToUse.systolic_bp || !dataToUse.fasting_glucose) {
      setError('Please provide at least Age, Blood Pressure, and Fasting Glucose to calculate CVD risk.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const fib4Num = parseFloat(calculateFIB4(dataToUse)) || 0;
      const payload = {
        age: parseFloat(dataToUse.age) || 0,
        sex: dataToUse.sex || 'Female',
        bmi: parseFloat(dataToUse.bmi) || 25.0,
        diabetes: dataToUse.diabetes || 'No',
        hypertension: dataToUse.hypertension || 'No',
        smoker: dataToUse.smoker || 'No',
        systolic_bp: parseFloat(dataToUse.systolic_bp) || 120,
        diastolic_bp: parseFloat(dataToUse.diastolic_bp) || 80,
        fasting_glucose: parseFloat(dataToUse.fasting_glucose) || 100,
        total_cholesterol: parseFloat(dataToUse.total_cholesterol) || 200,
        ldl: parseFloat(dataToUse.ldl) || 120,
        hdl: parseFloat(dataToUse.hdl) || 50,
        triglycerides: parseFloat(dataToUse.triglycerides) || 150,
        platelet_count: parseFloat(dataToUse.platelet_count) || 220,
        slcvd_score: parseFloat(dataToUse.slcvd_score) || 10.0,
        who_cvd_risk: parseFloat(dataToUse.who_cvd_risk) || 8.0,
        selected_model: modelToUse,
      };

      if (isFibrosis) {
        payload.ast = parseFloat(dataToUse.ast) || 25.0;
        payload.alt = parseFloat(dataToUse.alt) || 25.0;
        payload.fibrosis_stage = parseFloat(dataToUse.fibrosis_stage) || 0;
        payload.fib_4_score = fib4Num;
      }

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || errData.error || 'Prediction calculation failed');
      }

      const data = await res.json();
      if (targetMode === 'user') {
        setUserPrediction(data);
      } else {
        setPresetPrediction(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fib4Val = calculateFIB4(currentFormData);
  const fib4IsNumeric = fib4Val !== '—';
  const fib4IsHigh = fib4IsNumeric && parseFloat(fib4Val) >= 2.67;
  const fib4IsMod = fib4IsNumeric && parseFloat(fib4Val) >= 1.3 && parseFloat(fib4Val) < 2.67;

  const prob = currentPrediction ? parseFloat(currentPrediction.probability) || 0 : 0;
  // Needle angle calibrated to the clinical gauge ranges:
  // Green Arc (Low <10%): 0% to 10% -> -90° to -54°
  // Amber Arc (Moderate 10-20%): 10% to 20% -> -54° to -18°
  // Red Arc (High ≥20%): 20% to 50%+ -> -18° to +90°
  const needleAngle = (() => {
    if (prob <= 0) return -90;
    const clamped = Math.min(prob, 50);
    return -90 + (clamped / 50) * 180;
  })();

  return (
    <div className="clinical-workspace-rows">
      {/* TOP ROW: Patient Diagnostic Profile */}
      <div className="card" id="patientDiagnosticProfile">
        <div className="card-header-bar">
          <div className="header-icon-box terracotta">
            <User size={18} strokeWidth={2.4} />
          </div>
          <div className="card-header-text">
            <span className="card-eyebrow-tag">STEP 1 · DATA INPUT · CLINICAL PROFILE</span>
            <h2 className="card-title">Patient Diagnostic Profile</h2>
          </div>
        </div>

        {/* Top Controls Strip: Model Select at Front + Two-Way Input Mode Selector */}
        <div className="front-controls-strip">
          {/* 1. Model Selector at Front */}
          <div className="segmented-control-wrapper model-front-wrapper">
            <span className="segmented-control-label">Select Prediction Model:</span>
            <div className="clean-segmented-control" role="tablist">
              <button
                type="button"
                className={`segmented-pill-btn model-pill ${
                  isFibrosisModel ? 'active-model-b' : ''
                }`}
                onClick={() => {
                  const m = 'Model B (Fibrosis-Enhanced Logistic Regression)';
                  setSelectedModel(m);
                  if (currentPrediction) {
                    handleCalculateRisk(m);
                  }
                }}
                role="tab"
                aria-selected={isFibrosisModel}
                title="Model B includes Liver Fibrosis Biomarkers (20 features) - Recommended"
              >
                <Layers size={14} strokeWidth={2.4} />
                <span>Model B (Fibrosis-Enhanced Logistic Regression)</span>
                <span className="badge-pill-rec">★ Recommended</span>
              </button>
              <button
                type="button"
                className={`segmented-pill-btn model-pill ${
                  !isFibrosisModel ? 'active-model-a' : ''
                }`}
                onClick={() => {
                  const m = 'Model A (Baseline Traditional Logistic Regression)';
                  setSelectedModel(m);
                  if (currentPrediction) {
                    handleCalculateRisk(m);
                  }
                }}
                role="tab"
                aria-selected={!isFibrosisModel}
                title="Model A evaluates 16 traditional cardiovascular risk factors only"
              >
                <Sliders size={14} strokeWidth={2.4} />
                <span>Model A (Baseline Traditional Logistic Regression)</span>
                <span className="badge-pill-sub">16 Features</span>
              </button>
            </div>
          </div>

          {/* 2. Two-Way Input Mode Selector: User Input Values (Main) vs. Preloaded Reference Values */}
          <div className="segmented-control-wrapper mode-front-wrapper">
            <span className="segmented-control-label">Data Input Mode:</span>
            <div className="clean-segmented-control" role="tablist">
              <button
                type="button"
                className={`segmented-pill-btn ${inputMode === 'user' ? 'active' : ''}`}
                onClick={() => handleSwitchMode('user')}
                role="tab"
                aria-selected={inputMode === 'user'}
                title="Direct Patient Input: Enter your patient's specific clinical parameters (Main)"
              >
                <UserCheck size={14} strokeWidth={2.4} />
                <span>User Input Values</span>
                <span className="badge-pill-main">★ Main</span>
              </button>
              <button
                type="button"
                className={`segmented-pill-btn ${inputMode === 'preset' ? 'active' : ''}`}
                onClick={() => handleSwitchMode('preset')}
                role="tab"
                aria-selected={inputMode === 'preset'}
                title="Preloaded Reference Cases from Sri Lankan MASLD Registry"
              >
                <FolderOpen size={14} strokeWidth={2.4} />
                <span>Preloaded Reference Values</span>
                <span className="badge-pill-sub">3 Cases</span>
              </button>
            </div>
          </div>

          {/* 3. Sub-strip tailored to active Input Mode */}
          {inputMode === 'preset' ? (
            <div className="preset-sub-strip">
              <span className="sub-strip-label">Select Reference Case:</span>
              <div className="clean-segmented-control" role="tablist">
                <button
                  type="button"
                  className={`segmented-pill-btn ${activePreset === 'F3' ? 'active' : ''}`}
                  onClick={() => loadPreset('F3')}
                  role="tab"
                  aria-selected={activePreset === 'F3'}
                >
                  <span className="badge-dot red"></span>
                  <span>Case 1: F3 Severe</span>
                </button>
                <button
                  type="button"
                  className={`segmented-pill-btn ${activePreset === 'F2' ? 'active' : ''}`}
                  onClick={() => loadPreset('F2')}
                  role="tab"
                  aria-selected={activePreset === 'F2'}
                >
                  <span className="badge-dot amber"></span>
                  <span>Case 2: F2 Moderate</span>
                </button>
                <button
                  type="button"
                  className={`segmented-pill-btn ${activePreset === 'F0' ? 'active' : ''}`}
                  onClick={() => loadPreset('F0')}
                  role="tab"
                  aria-selected={activePreset === 'F0'}
                >
                  <span className="badge-dot green"></span>
                  <span>Case 3: F0 Low Risk</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="user-input-sub-strip">
              <span className="sub-strip-note">
                Direct Patient Input Mode active. Enter patient details or use quick actions:
              </span>
              <div className="user-input-quick-actions">
                <button
                  type="button"
                  className="quick-action-btn"
                  onClick={handleFillBaseline}
                  title="Pre-fill typical physiological baseline values to customize"
                >
                  <Sparkles size={12} />
                  <span>Fill Standard Baseline</span>
                </button>
                <button
                  type="button"
                  className="quick-action-btn text-muted"
                  onClick={handleClearFields}
                  title="Clear all fields to enter fresh patient data"
                >
                  <RotateCcw size={12} />
                  <span>Clear All</span>
                </button>
              </div>
            </div>
          )}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleCalculateRisk();
          }}
        >
          {/* TWO COLUMNS OF CARDS: Left Card (Demographics & Hemodynamics) & Right Card (Labs & Liver Fibrosis) */}
          <div className="profile-two-columns-grid">
            {/* LEFT COLUMN CARD: Demographics & Hemodynamics */}
            <div className="profile-column-card">
              <div className="column-card-header">
                <span className="group-num">1</span>
                <div>
                  <h3 className="column-card-title">Demographic &amp; Hemodynamic Profile</h3>
                  <span className="section-sub">Patient characteristics, anthropometrics, and blood pressure</span>
                </div>
              </div>

              {/* Section A */}
              <div className="group-box">
                <div className="group-title">
                  <span className="group-num">A</span>
                  Demographic &amp; Anthropometrics
                </div>
                <div className="input-grid cols-3">
                  <div className="input-field">
                    <label>Age (years)</label>
                    <input
                      type="number"
                      name="age"
                      value={currentFormData.age}
                      onChange={handleInputChange}
                      min="18"
                      max="100"
                      placeholder="e.g. 52"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>Sex</label>
                    <select name="sex" value={currentFormData.sex} onChange={handleInputChange}>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                  </div>
                  <div className="input-field">
                    <label>BMI (kg/m²)</label>
                    <input
                      type="number"
                      name="bmi"
                      step="0.1"
                      value={currentFormData.bmi}
                      onChange={handleInputChange}
                      placeholder="e.g. 28.4"
                      required
                    />
                    <span className="unit-note">Asian cutoff: ≥27.5 Obese</span>
                  </div>
                </div>
              </div>

              {/* Section B */}
              <div className="group-box">
                <div className="group-title">
                  <span className="group-num">B</span>
                  Clinical &amp; Hemodynamic Factors
                </div>
                <div className="input-grid cols-3">
                  <div className="input-field">
                    <label>Type 2 Diabetes</label>
                    <select name="diabetes" value={currentFormData.diabetes} onChange={handleInputChange}>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="input-field">
                    <label>Hypertension</label>
                    <select name="hypertension" value={currentFormData.hypertension} onChange={handleInputChange}>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                  <div className="input-field">
                    <label>Smoker</label>
                    <select name="smoker" value={currentFormData.smoker} onChange={handleInputChange}>
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                    </select>
                  </div>
                </div>
                <div className="input-grid cols-2" style={{ marginTop: 12 }}>
                  <div className="input-field">
                    <label>Systolic BP (mmHg)</label>
                    <input
                      type="number"
                      name="systolic_bp"
                      value={currentFormData.systolic_bp}
                      onChange={handleInputChange}
                      placeholder="e.g. 135"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>Diastolic BP (mmHg)</label>
                    <input
                      type="number"
                      name="diastolic_bp"
                      value={currentFormData.diastolic_bp}
                      onChange={handleInputChange}
                      placeholder="e.g. 85"
                      required
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN CARD: Laboratory & Liver Fibrosis */}
            <div className="profile-column-card">
              <div className="column-card-header">
                <span className="group-num teal">2</span>
                <div>
                  <h3 className="column-card-title">Laboratory Parameters &amp; Liver Fibrosis</h3>
                  <span className="section-sub">Lipid panel, metabolic indicators, and hepatic staging</span>
                </div>
              </div>

              {/* Section C */}
              <div className="group-box">
                <div className="group-title">
                  <span className="group-num">C</span>
                  Laboratory Parameters &amp; Lipid Profile
                </div>
                <div className="input-grid cols-3">
                  <div className="input-field">
                    <label>Fasting Glucose (mg/dL)</label>
                    <input
                      type="number"
                      name="fasting_glucose"
                      value={currentFormData.fasting_glucose}
                      onChange={handleInputChange}
                      placeholder="e.g. 110"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>Total Cholesterol (mg/dL)</label>
                    <input
                      type="number"
                      name="total_cholesterol"
                      value={currentFormData.total_cholesterol}
                      onChange={handleInputChange}
                      placeholder="e.g. 205"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>LDL-C (mg/dL)</label>
                    <input
                      type="number"
                      name="ldl"
                      value={currentFormData.ldl}
                      onChange={handleInputChange}
                      placeholder="e.g. 130"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>HDL-C (mg/dL)</label>
                    <input
                      type="number"
                      name="hdl"
                      value={currentFormData.hdl}
                      onChange={handleInputChange}
                      placeholder="e.g. 48"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>Triglycerides (mg/dL)</label>
                    <input
                      type="number"
                      name="triglycerides"
                      value={currentFormData.triglycerides}
                      onChange={handleInputChange}
                      placeholder="e.g. 165"
                      required
                    />
                  </div>
                  <div className="input-field">
                    <label>Platelet Count (10⁹/L)</label>
                    <input
                      type="number"
                      name="platelet_count"
                      value={currentFormData.platelet_count}
                      onChange={handleInputChange}
                      placeholder="e.g. 220"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Section D: Key Liver Biomarkers (Dynamically displayed ONLY for Fibrosis-Enhanced model) */}
              {isFibrosisModel ? (
                <div className="group-box highlight-teal">
                  <div className="group-title">
                    <span className="group-num teal">D</span>
                    Liver Fibrosis Severity (Key Novel Biomarker)
                  </div>
                  <div className="input-grid cols-3">
                    <div className="input-field">
                      <label>Fibrosis Stage</label>
                      <select
                        name="fibrosis_stage"
                        className="select-teal"
                        value={currentFormData.fibrosis_stage}
                        onChange={handleInputChange}
                      >
                        <option value="0">F0 - None (Healthy / Simple Steatosis)</option>
                        <option value="1">F1 - Mild Periportal Fibrosis</option>
                        <option value="2">F2 - Moderate / Bridging Fibrosis</option>
                        <option value="3">F3 - Severe / Advanced Fibrosis</option>
                        <option value="4">F4 - Cirrhosis</option>
                      </select>
                    </div>
                    <div className="input-field">
                      <label>AST (U/L)</label>
                      <input
                        type="number"
                        name="ast"
                        value={currentFormData.ast}
                        onChange={handleInputChange}
                        placeholder="e.g. 35"
                        required
                      />
                    </div>
                    <div className="input-field">
                      <label>ALT (U/L)</label>
                      <input
                        type="number"
                        name="alt"
                        value={currentFormData.alt}
                        onChange={handleInputChange}
                        placeholder="e.g. 45"
                        required
                      />
                    </div>
                  </div>

                  {/* Live FIB-4 Computed Card */}
                  <div style={{ marginTop: 12 }}>
                    <div className="fib4-metric-card">
                      <span style={{ fontSize: '11px', fontWeight: 700, color: '#0f766e' }}>
                        AUTO-CALCULATED FIB-4:
                      </span>
                      <span className="fib4-score-text">{fib4Val}</span>
                      <span className={`fib4-tag ${fib4IsHigh ? 'high' : fib4IsMod ? 'mod' : 'low'}`}>
                        {fib4Val === '—'
                          ? 'Awaiting AST & ALT'
                          : fib4IsHigh
                          ? 'High Risk (>2.67)'
                          : fib4IsMod
                          ? 'Intermediate (1.30-2.67)'
                          : 'Low Risk (<1.30)'}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="group-box-notice">
                  <Sliders size={18} />
                  <div>
                    <strong>Section D (Liver Fibrosis Biomarkers) Hidden:</strong> The active model (<strong>Model A: Baseline Traditional Logistic Regression</strong>) evaluates cardiovascular risk exclusively using 16 traditional demographic, hemodynamic, and lipid biomarkers. AST, ALT, Fibrosis Stage, and FIB-4 are not utilized. Switch to <strong>Model B (Fibrosis-Enhanced Logistic Regression)</strong> above to evaluate liver fibrosis features.
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Submit Action (Full Width across bottom of Profile card) */}
          <button type="submit" className="cta-calculate-btn terracotta-btn" disabled={loading}>
            <Sparkles size={18} strokeWidth={2.2} />
            <span>{loading ? 'Evaluating Risk & Running SHAP...' : 'Calculate Cardiovascular Risk & Explain'}</span>
          </button>
        </form>
      </div>

      {/* BOTTOM ROW: AI Decision Support Report directly below */}
      <div className="card" id="clinicalReport">
        <div className="card-header-bar">
          <div className="header-icon-box terracotta">
            <Activity size={18} strokeWidth={2.4} />
          </div>
          <div className="card-header-text">
            <span className="card-eyebrow-tag">STEP 2 · AI DECISION SUPPORT REPORT</span>
            <h2 className="card-title">Patient-Specific Risk &amp; SHAP Explanation</h2>
          </div>
        </div>

        {error && (
          <div className="error-alert" style={{ marginBottom: 16 }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* Clean Empty State when awaiting patient input */}
        {!currentPrediction && !loading && (
          <div className="empty-state-view">
            <div className="pulse-orbit">
              <Activity size={32} strokeWidth={2.2} />
            </div>
            <h3>Direct Patient Evaluation Mode</h3>
            <p>
              Enter patient diagnostic parameters in the profile above and click{' '}
              <strong>"Calculate Cardiovascular Risk &amp; Explain"</strong> to generate the 10-year CVD risk score,
              model comparison, and SHAP feature attribution.
            </p>
            <div style={{ marginTop: 16 }}>
              <button
                type="button"
                className="quick-action-btn"
                style={{ padding: '6px 14px', fontSize: '12px' }}
                onClick={handleFillBaseline}
              >
                <Sparkles size={14} />
                <span>Load Standard Baseline Template</span>
              </button>
            </div>
          </div>
        )}

        {currentPrediction && (
          <div>
            {/* 1. TOP SECTION: Patient-Level Explanation & Fibrosis Contribution (As in Clinical Design) */}
            <div className="patient-explanation-grid">
              {/* CARD 1: Patient-Level Explanation */}
              <div className="explanation-card">
                <div className="explanation-card-header">
                  <h3 className="explanation-card-title">Patient-Level Explanation</h3>
                  <div className="explanation-card-subtitle">
                    Why did the model give this prediction for this patient?
                  </div>
                </div>

                <div className="waterfall-header-row">
                  <span>Predictive Factors</span>
                  <span>SHAP Impact</span>
                </div>

                <div className="waterfall-items-list">
                  {/* Base Value (Avg. Risk) */}
                  <div className="waterfall-item-row">
                    <span className="waterfall-factor-name base-name">Base Value (Avg. Risk)</span>
                    <div className="waterfall-bar-track">
                      <div
                        className="waterfall-bar-block base-block"
                        style={{ width: `${Math.min(((currentPrediction.base_value || 0.22) / 0.5) * 80, 85)}%` }}
                      ></div>
                    </div>
                    <span className={`waterfall-impact-val ${(currentPrediction.base_value || 0) >= 0 ? 'pos' : 'neg'}`}>
                      {(currentPrediction.base_value || 0.22) >= 0 ? '+' : ''}{(currentPrediction.base_value || 0.22).toFixed(2)}
                    </span>
                  </div>

                  {/* Top Patient Predictive Factors */}
                  {currentPrediction.shap_explanations.slice(0, 6).map((item, idx) => {
                    const isPos = item.is_risk_increasing;
                    const absVal = Math.abs(item.shap_value);
                    const barWidth = Math.min(Math.max(absVal * 180, 20), 92);
                    const isFib = item.is_fibrosis;

                    return (
                      <div key={idx} className="waterfall-item-row">
                        <span className="waterfall-factor-name" title={item.feature}>
                          {getFriendlyFactorName(item)}
                        </span>
                        <div className="waterfall-bar-track">
                          <div
                            className={`waterfall-bar-block ${
                              isFib
                                ? 'coral-block'
                                : isPos
                                ? 'blue-block'
                                : 'green-block'
                            }`}
                            style={{ width: `${barWidth}%` }}
                          ></div>
                        </div>
                        <span className={`waterfall-impact-val ${isFib ? 'high-pos' : isPos ? 'pos' : 'neg'}`}>
                          {isPos ? '+' : ''}{item.shap_value.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>

                {/* Final Prediction Box */}
                <div className="final-prediction-box">
                  <div>
                    <div className="final-pred-label">Final Prediction</div>
                    <span className="final-pred-sub">(Patient Risk)</span>
                  </div>
                  <div className="final-pred-val">
                    {(currentPrediction.probability / 100).toFixed(2)}{' '}
                    <span style={{ fontSize: '18px', fontWeight: 700 }}>
                      ({currentPrediction.probability}%)
                    </span>
                  </div>
                </div>
              </div>

              {/* CARD 2: Fibrosis Contribution */}
              <div className="explanation-card">
                <div className="explanation-card-header">
                  <h3 className="explanation-card-title">Fibrosis Contribution</h3>
                  <div className="explanation-card-subtitle">
                    How much did fibrosis severity contribute to the risk?
                  </div>
                </div>

                {/* Donut Graphic */}
                <div className="fibrosis-donut-wrap">
                  <div className="fibrosis-donut-graphic">
                    <svg viewBox="0 0 160 160" width="175" height="175">
                      {/* Background circle */}
                      <circle
                        cx="80"
                        cy="80"
                        r="58"
                        fill="none"
                        stroke="#f1f5f9"
                        strokeWidth="15"
                      />
                      {/* Segment 1: Lavender Base Arc */}
                      <circle
                        cx="80"
                        cy="80"
                        r="58"
                        fill="none"
                        stroke="#e0e7ff"
                        strokeWidth="15"
                        strokeDasharray="364.4"
                        strokeDashoffset="0"
                      />
                      {/* Segment 2: Amber Metabolic Arc */}
                      <circle
                        cx="80"
                        cy="80"
                        r="58"
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="15"
                        strokeDasharray="140 224.4"
                        strokeDashoffset="-70"
                      />
                      {/* Segment 3: Red Fibrosis Arc */}
                      <circle
                        cx="80"
                        cy="80"
                        r="58"
                        fill="none"
                        stroke={isFibrosisModel ? '#e11d48' : '#94a3b8'}
                        strokeWidth="15"
                        strokeDasharray={`${Math.max((currentPrediction.fibrosis_contribution_pct / 100) * 364.4, 25)} 364.4`}
                        strokeDashoffset="-210"
                        strokeLinecap="round"
                      />
                    </svg>

                    {/* Donut Center with Liver Icon */}
                    <div className="fibrosis-donut-center">
                      <svg viewBox="0 0 64 64" width="24" height="24" fill="none">
                        <path
                          d="M14 26 C14 18, 26 14, 38 18 C50 22, 54 32, 52 42 C50 50, 36 54, 26 50 C18 46, 14 36, 14 26 Z"
                          fill="#991b1b"
                          stroke="#7f1d1d"
                          strokeWidth="1.5"
                        />
                        <path
                          d="M26 22 C34 22, 44 26, 48 34"
                          stroke="#fca5a5"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M34 32 C38 34, 44 40, 42 46"
                          stroke="#fca5a5"
                          strokeWidth="2"
                          strokeLinecap="round"
                        />
                        <circle cx="28" cy="40" r="3.5" fill="#15803d" />
                      </svg>

                      <span className="fibrosis-center-caption">Fibrosis Contribution</span>
                      <span className="fibrosis-center-percent">
                        {isFibrosisModel ? `${currentPrediction.fibrosis_contribution_pct}%` : '0%'}
                      </span>
                      <span className="fibrosis-center-tag">
                        {isFibrosisModel
                          ? `(${currentPrediction.fibrosis_significance || 'Active Shift'})`
                          : '(Model A Traditional Only)'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Detailed Clinical Explanation Based on Inputs and Output */}
                <div className="fibrosis-clinical-narrative">
                  <div className="narrative-title">
                    <ShieldCheck size={14} color="#0d9488" />
                    <span>Clinical Risk Attribution Breakdown</span>
                  </div>

                  <div className="narrative-bullet">
                    <strong>Patient Inputs:</strong>
                    <span>
                      {isFibrosisModel ? (
                        <>
                          Fibrosis Stage <strong>F{currentFormData.fibrosis_stage || 0}</strong> (FIB-4: <strong>{fib4Val}</strong>, AST: <strong>{currentFormData.ast || 25} U/L</strong>, ALT: <strong>{currentFormData.alt || 25} U/L</strong>), BP: <strong>{currentFormData.systolic_bp}/{currentFormData.diastolic_bp} mmHg</strong>, Fasting Glucose: <strong>{currentFormData.fasting_glucose} mg/dL</strong>, Diabetes: <strong>{currentFormData.diabetes}</strong>.
                        </>
                      ) : (
                        <>
                          Evaluated with 16 traditional risk factors: Age <strong>{currentFormData.age} yrs</strong>, BP <strong>{currentFormData.systolic_bp}/{currentFormData.diastolic_bp} mmHg</strong>, Fasting Glucose <strong>{currentFormData.fasting_glucose} mg/dL</strong>, Diabetes: <strong>{currentFormData.diabetes}</strong>.
                        </>
                      )}
                    </span>
                  </div>

                  <div className="narrative-bullet">
                    <strong>Output &amp; Shift:</strong>
                    <span>
                      {isFibrosisModel ? (
                        <>
                          10-Year CVD Risk is <strong>{currentPrediction.probability}%</strong> ({currentPrediction.risk_category} Risk). Including liver fibrosis adds <strong>{currentPrediction.fibrosis_risk_shift >= 0 ? '+' : ''}{currentPrediction.fibrosis_risk_shift} pp</strong> risk shift compared to traditional Framingham factors alone (<strong>{currentPrediction.baseline_probability}%</strong> &rarr; <strong>{currentPrediction.enhanced_probability || currentPrediction.probability}%</strong>).
                        </>
                      ) : (
                        <>
                          10-Year CVD Risk is <strong>{currentPrediction.probability}%</strong> ({currentPrediction.risk_category} Risk). Fibrosis contribution is <strong>0% / Not Evaluated</strong> by Model A.
                        </>
                      )}
                    </span>
                  </div>

                  <div className="narrative-summary-text">
                    {isFibrosisModel ? (
                      parseInt(currentFormData.fibrosis_stage || 0) >= 3 ? (
                        <>
                          <strong>Hepatic Mechanism (F3-F4 Advanced Fibrosis):</strong> Advanced bridging fibrosis acts as an independent cardiovascular risk multiplier. Severe hepatic vascular remodeling and persistent inflammation accelerate the systemic release of pro-atherogenic cytokines (IL-6, TNF-&alpha;) and asymmetric dimethylarginine (ADMA), driving premature endothelial dysfunction and coronary plaque destabilization. This explains why hepatic biomarkers account for <strong>{currentPrediction.fibrosis_contribution_pct}%</strong> of this patient's total CVD risk.
                        </>
                      ) : parseInt(currentFormData.fibrosis_stage || 0) === 2 ? (
                        <>
                          <strong>Hepatic Mechanism (F2 Moderate Fibrosis):</strong> Significant periportal and sinusoidal fibrosis exacerbates atherogenic lipid fractions and activates systemic microvascular inflammatory signaling, contributing a notable <strong>{currentPrediction.fibrosis_contribution_pct}%</strong> independent risk shift.
                        </>
                      ) : (
                        <>
                          <strong>Hepatic Mechanism (F0-F1 Mild / None):</strong> Preserved hepatic architecture contributes minimally (<strong>{currentPrediction.fibrosis_contribution_pct}%</strong>) to cardiovascular risk. This patient's 10-year CVD risk is predominantly governed by traditional cardiometabolic determinants (blood pressure, lipid profile, and glycemic control).
                        </>
                      )
                    ) : (
                      <>
                        <strong>Traditional Model Scope:</strong> Model A calculates cardiovascular risk exclusively from 16 traditional risk factors. Switch to <strong>Model B (Fibrosis-Enhanced)</strong> above to incorporate hepatic fibrosis staging, FIB-4 index, and aminotransferase kinetics.
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* 2. Comprehensive Model Comparison & Gauge Metrics */}
            <div className="report-metrics-row">
              {/* Main Risk Gauge Card */}
              <div className="risk-highlight-card">
                <div>
                  <div className="risk-badge-row">
                    <span
                      className={`risk-status-pill ${
                        currentPrediction.risk_category === 'High'
                          ? 'danger'
                          : currentPrediction.risk_category === 'Moderate'
                          ? 'warning'
                          : 'success'
                      }`}
                    >
                      {currentPrediction.risk_category.toUpperCase()} RISK
                    </span>
                  </div>
                  <div className="probability-headline">
                    {currentPrediction.probability}
                    <span className="prob-pct">%</span>
                  </div>
                  <div className="prob-subtext">Estimated 10-Year Major Adverse Cardiovascular Event (MACE) Risk</div>
                </div>

                {/* Semi-Circular SVG Precision Gauge calibrated to clinical risk ranges */}
                <div className="gauge-display-wrap">
                  <svg viewBox="0 0 200 115" className="precision-gauge-svg">
                    {/* Green Arc: Low Risk (<10%) - 0% to 10% */}
                    <path d="M 20 100 A 80 80 0 0 1 34.5 54.1" fill="none" stroke="#10b981" strokeWidth="14" strokeLinecap="round" />
                    {/* Amber Arc: Moderate Risk (10-20%) - 10% to 20% */}
                    <path d="M 36.5 51.3 A 80 80 0 0 1 73.3 24.6" fill="none" stroke="#f59e0b" strokeWidth="14" />
                    {/* Red Arc: High Risk (≥20%) - 20% to 50%+ */}
                    <path d="M 77.3 23.3 A 80 80 0 0 1 180 100" fill="none" stroke="#ef4444" strokeWidth="14" strokeLinecap="round" />

                    {/* Center Pivot & Needle */}
                    <circle cx="100" cy="100" r="7" fill="#0f172a" />
                    <line
                      x1="100"
                      y1="100"
                      x2="100"
                      y2="28"
                      stroke="#0f172a"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      style={{
                        transformOrigin: '100px 100px',
                        transform: `rotate(${needleAngle}deg)`,
                        transition: 'transform 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
                      }}
                    />
                  </svg>
                  <div className="gauge-scale-legend">
                    <span className="scale-item green">Low (&lt;10%)</span>
                    <span className="scale-item amber">Moderate (10-20%)</span>
                    <span className="scale-item red">High (≥20%)</span>
                  </div>
                </div>
              </div>

              {/* Model Comparison: Model A vs. Model B */}
              <div className="comparison-strip">
                <div className={`comp-box ${!isFibrosisModel ? 'highlighted' : ''}`}>
                  <span className="comp-label">MODEL A · BASELINE (16 FEATURES)</span>
                  <span className="comp-val mono">{currentPrediction.baseline_probability}%</span>
                  <span className="comp-sub">Logistic Regression · Traditional CVD (AUC = 0.851)</span>
                </div>
                <div className={`comp-box ${isFibrosisModel ? 'highlighted' : ''}`}>
                  <span className="comp-label">MODEL B · FIBROSIS-ENHANCED (20 FEATURES)</span>
                  <span className="comp-val mono">{currentPrediction.enhanced_probability || currentPrediction.probability}%</span>
                  <span className="comp-sub">Logistic Regression + Liver Fibrosis (AUC = 0.881)</span>
                </div>
                <div className="comp-box donut-box">
                  <div className="mini-donut">
                    <span className="donut-figure mono">{currentPrediction.fibrosis_contribution_pct}%</span>
                  </div>
                  <div>
                    <span className="comp-label">FIBROSIS CONTRIBUTION</span>
                    <span className="donut-status">
                      {currentPrediction.fibrosis_significance} ({currentPrediction.fibrosis_risk_shift >= 0 ? '+' : ''}
                      {currentPrediction.fibrosis_risk_shift} pp)
                    </span>
                    <span className="comp-sub">Fraction of total risk explained by liver fibrosis</span>
                  </div>
                </div>
              </div>
            </div>

            {/* 3. Evidence-Based Clinical Recommendations */}
            <div className="clinical-recs-module" style={{ marginTop: 22, paddingTop: 20 }}>
              <div className="module-title-bar" style={{ justifyContent: 'flex-start' }}>
                <div className="title-left">
                  <ShieldCheck size={18} strokeWidth={2.3} style={{ color: 'var(--primary-teal)', flexShrink: 0 }} />
                  <div>
                    <h3 className="module-title">Evidence-Based Clinical Decision Support Recommendations</h3>
                    <span className="module-desc">Actionable clinical guidance derived from risk level and fibrosis stage</span>
                  </div>
                </div>
              </div>

              <div className="recs-card-grid">
                {currentPrediction.recommendations.map((rec, i) => (
                  <div key={i} className={`rec-item ${rec.category.includes('Cardio') ? 'priority' : ''}`}>
                    <div className="rec-cat">{rec.category}</div>
                    <div className="rec-head">{rec.action}</div>
                    <div className="rec-text">{rec.detail}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
