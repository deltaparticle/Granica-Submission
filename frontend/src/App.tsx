import { useState, useEffect } from 'react';
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
}

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
  const [comparison, setComparison] = useState<ComparisonData | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasets(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch datasets:", err);
        setError("Failed to connect to the API. Is the FastAPI backend running?");
        setLoading(false);
      });
  }, []);

  const handleCardClick = (datasetKey: string) => {
    setSelectedDataset(datasetKey);
    setComparisonLoading(true);
    setComparison(null);

    fetch(`${API_BASE}/api/compare/${datasetKey}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch comparison data");
        return res.json();
      })
      .then(data => {
        setComparison(data);
        setComparisonLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch comparison:", err);
        setComparisonLoading(false);
      });
  };

  const closeModal = () => {
    setSelectedDataset(null);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Canary</h1>
        <p>Zero-Shot Multimodal Fault Detection - Spectrogram Explorer</p>
      </header>

      {error && (
        <div style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', padding: '1rem', borderRadius: '8px', color: '#f87171', textAlign: 'center', marginBottom: '2rem', border: '1px solid #ef4444' }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="loader">
          <div className="spinner"></div>
          <p>Loading Datasets...</p>
        </div>
      ) : (
        <div className="dashboard">
          {datasets.map(ds => (
            <div key={ds.key} className="card" onClick={() => handleCardClick(ds.key)}>
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
      )}

      {selectedDataset && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{comparison?.dataset_name || 'Loading Analysis...'}</h2>
              <button className="close-btn" onClick={closeModal}>&times;</button>
            </div>
            
            <div className="modal-body">
              {comparisonLoading ? (
                <div className="loader">
                  <div className="spinner"></div>
                  <p>Generating Spectrograms and Waveforms (this may take a few seconds)...</p>
                </div>
              ) : comparison ? (
                <div className="comparison-container">
                  
                  {/* Heatmap comparison panel */}
                  <div>
                    <h3 style={{ marginBottom: '1rem', fontSize: '1.25rem', color: '#e0e0e0' }}>Spectrogram Difference Analysis</h3>
                    <img 
                      src={`data:image/png;base64,${comparison.comparison_b64}`} 
                      alt="Comparison Heatmap" 
                      className="comparison-image"
                    />
                  </div>
                  
                  {/* Waveforms side-by-side */}
                  <div className="waveforms-grid">
                    <div className="waveform-card normal">
                      <h4 style={{ color: '#4ade80' }}>Normal Baseline</h4>
                      <img src={`data:image/png;base64,${comparison.waveform_normal_b64}`} alt="Normal Waveform" />
                    </div>
                    
                    <div className="waveform-card faulty">
                      <h4 style={{ color: '#f87171' }}>Fault Detected: {comparison.fault_detail}</h4>
                      <img src={`data:image/png;base64,${comparison.waveform_faulty_b64}`} alt="Faulty Waveform" />
                    </div>
                  </div>
                  
                </div>
              ) : (
                <div style={{ color: '#f87171', textAlign: 'center', padding: '2rem' }}>
                  Failed to load comparison data.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
