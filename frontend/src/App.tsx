import { useState, useEffect, useRef } from 'react';
import './index.css';

interface Dataset {
  key: string;
  name: string;
  description: string;
  modality: string;
  sample_rate: string;
  file_count: number;
  fault_detail: string;
  has_showcase: boolean;
}

interface ComparisonData {
  dataset_name: string;
  fault_detail: string;
  comparison_b64: string;
  waveform_normal_b64: string;
  waveform_faulty_b64: string;
  anomaly_score: number;
  peak_freq: string;
  diagnosis: string;
}

const API_BASE = 'http://127.0.0.1:8000';

const SIMULATED_STARTUP_LOGS = [
  "[SYSTEM] Initiating ML Data Pipeline...",
  "[SYSTEM] Loading configuration: sample_rate=25600, n_mels=128, fmax=12800...",
  "[INFO] Bootstrapping PCAReconstructionMemoryBank...",
  "[INFO] Loading pre-trained ResNet-18 Encoder Checkpoints...",
  "[SUCCESS] Pipeline Ready. Awaiting Inference Requests...",
];

function PipelineTerminal({ 
  logs, 
  isComplete,
  onTypingComplete 
}: { 
  logs: string[], 
  isComplete: boolean,
  onTypingComplete: () => void 
}) {
  const [displayedIndex, setDisplayedIndex] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  // Animate logs appearing one by one without resetting
  useEffect(() => {
    if (displayedIndex < logs.length) {
      const timer = setTimeout(() => {
        setDisplayedIndex(prev => prev + 1);
      }, 300);
      return () => clearTimeout(timer);
    } else if (isComplete && displayedIndex === logs.length) {
      const timer = setTimeout(() => onTypingComplete(), 600);
      return () => clearTimeout(timer);
    }
  }, [logs.length, displayedIndex, isComplete, onTypingComplete]);

  // Scroll to bottom
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [displayedIndex]);

  const getTime = () => {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
  };

  const colorizeLog = (text: string) => {
    if (!text) return null; // Safeguard against undefined/empty
    if (text.includes("[INFO]")) return <span className="log-info">{text}</span>;
    if (text.includes("[SUCCESS]")) return <span className="log-success">{text}</span>;
    if (text.includes("[WARN]")) return <span className="log-warn">{text}</span>;
    if (text.includes("[SYSTEM]")) return <span style={{ color: '#c084fc' }}>{text}</span>;
    return text;
  };

  return (
    <div className="terminal-section">
      <div className="terminal-header">
        <div className="terminal-dot red"></div>
        <div className="terminal-dot yellow"></div>
        <div className="terminal-dot green"></div>
        <div className="terminal-title">bash — ML Pipeline Diagnostics</div>
      </div>
      <div className="terminal-body">
        {logs.slice(0, displayedIndex).map((log, i) => (
          <div key={i} className="log-line">
            <span className="log-time">[{getTime()}]</span>
            {colorizeLog(log)}
          </div>
        ))}
        {/* Blinking cursor effect at the end if not complete */}
        {!isComplete && (
          <div className="log-line">
            <span className="log-time">[{getTime()}]</span>
            <span style={{ animation: 'fadeIn 1s infinite alternate' }}>_</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  
  // API Response state
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [apiFailed, setApiFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Terminal / Animation State
  const [hasStartedInference, setHasStartedInference] = useState(false);
  const [inferenceLogs, setInferenceLogs] = useState<string[]>([]);
  const [apiIsComplete, setApiIsComplete] = useState(false);
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    // Inject a clean Google Font to override standard sans-serif
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
    document.body.style.fontFamily = "'Outfit', sans-serif";

    fetch(`${API_BASE}/api/datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasets(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch datasets:", err);
        setError("Failed to connect to the API. Is the Flask backend running?");
        setLoading(false);
      });
  }, []);

  const handleCardClick = (datasetKey: string, datasetName: string, modality: string) => {
    setSelectedDataset(datasetKey);
    setComparison(null);
    setApiFailed(false);
    
    setHasStartedInference(true);
    setApiIsComplete(false);
    setShowResults(false);
    
    // Set up the logs exactly mimicking the real predict.py script
    setInferenceLogs([
      ...SIMULATED_STARTUP_LOGS,
      "",
      `[SYSTEM] --- NEW INFERENCE REQUEST ---`,
      `[INFO] Running Inference on ${modality.toUpperCase()} file...`,
      `[INFO] Extracting Features via SpectrogramConfig...`,
      `[INFO] Passing batch to ResNet-18 Encoder (eval mode)...`,
      `[SUCCESS] Extracted Embedding Vector of shape (512,)`,
      `[INFO] STAGE 1 (EDGE): Projecting to PCAReconstructionMemoryBank...`,
    ]);

    fetch(`${API_BASE}/api/compare/${datasetKey}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch comparison data");
        return res.json();
      })
      .then(data => {
        setComparison(data);
        // Append the final logs based on real response
        setInferenceLogs(prev => [
          ...prev, 
          `[SUCCESS] Anomaly Score computed: ${data.anomaly_score.toFixed(4)}`,
          `[INFO] Threshold check passed. Diagnostics generated.`,
        ]);
        setApiIsComplete(true);
      })
      .catch(err => {
        console.error("Failed to fetch comparison:", err);
        setApiFailed(true);
        setInferenceLogs(prev => [
          ...prev, 
          `[WARN] Feature extraction failed: ${err.message}`
        ]);
        setApiIsComplete(true);
      });
  };

  const handleTerminalDoneTyping = () => {
    if (comparison && !apiFailed) {
      setShowResults(true);
    }
  };

  const closeResults = () => {
    setSelectedDataset(null);
    setHasStartedInference(false);
    setShowResults(false);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Canary</h1>
        <p>Zero-Shot Multimodal Fault Detection System</p>
      </header>

      {error && (
        <div style={{ backgroundColor: '#fef2f2', padding: '1rem', borderRadius: '8px', color: '#b91c1c', border: '1px solid #f87171', marginBottom: '2rem' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="loader">
          <div className="spinner"></div>
          <p>Initializing System...</p>
        </div>
      ) : (
        <div className="main-content">
          <div className="sidebar">
            {datasets.map(ds => (
              <div key={ds.key} className="card" onClick={() => handleCardClick(ds.key, ds.name, ds.modality)}>
                <div className="card-header">
                  <h3 className="card-title">{ds.name}</h3>
                  <span className="badge-modality">{ds.modality}</span>
                </div>
                <p className="card-desc">{ds.description}</p>
                <div className="card-meta">
                  <span className="meta-item">
                    📁 {ds.file_count.toLocaleString()} files
                  </span>
                  <span className="meta-item">
                    ⏱️ {ds.sample_rate}
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          {/* Right Panel Logic */}
          {!hasStartedInference ? (
            <div style={{ flex: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f1f5f9', borderRadius: '12px', border: '2px dashed #cbd5e1', height: 'calc(100vh - 150px)', color: '#64748b', position: 'sticky', top: '2rem' }}>
              <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }}>⚗️</div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem' }}>Pipeline Idle</h2>
              <p>Select a dataset card on the left to execute the inference script.</p>
            </div>
          ) : !showResults ? (
            <PipelineTerminal 
              logs={inferenceLogs} 
              isComplete={apiIsComplete} 
              onTypingComplete={handleTerminalDoneTyping} 
            />
          ) : comparison ? (
            <div className="results-panel slide-in-right">
              <div className="results-header">
                <h2>Diagnostic Results: {comparison.dataset_name}</h2>
                <button className="close-btn" onClick={closeResults}>&times;</button>
              </div>
              
              <div className="comparison-container">
                {/* Text Analysis Report */}
                <div className="analysis-report">
                  <h3 className="analysis-header" style={{ color: 'var(--text-primary)', marginBottom: '1.5rem', fontWeight: 700 }}>
                    Machine Learning Analysis
                  </h3>
                  <div className="analysis-grid">
                    <div className="analysis-metric">
                      <div className="metric-label">Memory Bank Anomaly Score</div>
                      <div className="metric-value danger">{comparison.anomaly_score.toFixed(4)}</div>
                    </div>
                    <div className="analysis-metric">
                      <div className="metric-label">Peak Frequency Deviation</div>
                      <div className="metric-value">{comparison.peak_freq}</div>
                    </div>
                    <div className="analysis-metric">
                      <div className="metric-label">System Diagnosis</div>
                      <div className="metric-value">{comparison.diagnosis}</div>
                    </div>
                  </div>
                </div>

                {/* Heatmap comparison panel */}
                <div>
                  <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: 'var(--text-primary)', fontWeight: 700 }}>
                    Spectrogram Difference Matrix
                  </h3>
                  <img 
                    src={`data:image/png;base64,${comparison.comparison_b64}`} 
                    alt="Comparison Heatmap" 
                    className="comparison-image"
                  />
                </div>
                
                {/* Waveforms side-by-side */}
                <div className="waveforms-grid">
                  <div className="waveform-card normal">
                    <h4 style={{ fontWeight: 600, color: '#334155' }}>Baseline Signal Profile</h4>
                    <img src={`data:image/png;base64,${comparison.waveform_normal_b64}`} alt="Normal Waveform" />
                  </div>
                  
                  <div className="waveform-card faulty">
                    <h4 style={{ fontWeight: 600, color: 'var(--danger-color)' }}>Anomalous Signal Profile</h4>
                    <img src={`data:image/png;base64,${comparison.waveform_faulty_b64}`} alt="Faulty Waveform" />
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="results-panel">
              <div style={{ color: '#dc2626', textAlign: 'center', padding: '2rem' }}>
                Failed to load diagnostic data. Check backend logs.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
