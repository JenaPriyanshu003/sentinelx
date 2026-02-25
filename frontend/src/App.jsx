import { useState, useRef, useCallback } from 'react';
import './App.css';

// Inline SVG logo icon
const ShieldIcon = () => (
  <svg className="logo-icon" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M19 3L5 9v10c0 8.3 5.9 16.1 14 18 8.1-1.9 14-9.7 14-18V9L19 3z"
      stroke="#00d4ff" strokeWidth="2" fill="rgba(0,212,255,0.08)" />
    <path d="M13 19l4 4 8-8" stroke="#00d4ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

// Inline SVG upload icon
const UploadIcon = () => (
  <svg className="upload-icon-svg" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M16 4v18M8 12l8-8 8 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M6 24h20" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
);

// Inline hero SVG background
const HeroBg = () => (
  <svg className="hero-svg-bg" viewBox="0 0 1100 260" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
    <rect width="1100" height="260" fill="#050b18" />
    {/* Grid lines */}
    {Array.from({ length: 20 }).map((_, i) => (
      <line key={`v${i}`} x1={i * 60} y1="0" x2={i * 60} y2="260" stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
    ))}
    {Array.from({ length: 6 }).map((_, i) => (
      <line key={`h${i}`} x1="0" y1={i * 52} x2="1100" y2={i * 52} stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
    ))}
    {/* Glow circles */}
    <circle cx="750" cy="130" r="180" fill="rgba(0,212,255,0.04)" />
    <circle cx="850" cy="80" r="80" fill="rgba(139,92,246,0.05)" />
    {/* Face scan outline */}
    <ellipse cx="820" cy="130" rx="60" ry="75" stroke="rgba(0,212,255,0.15)" strokeWidth="1.5" fill="none" />
    <circle cx="800" cy="115" r="5" fill="rgba(0,212,255,0.3)" />
    <circle cx="840" cy="115" r="5" fill="rgba(0,212,255,0.3)" />
    <path d="M805 148 Q820 158 835 148" stroke="rgba(0,212,255,0.3)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
    {/* Corner brackets */}
    <path d="M760 55 h15 M760 55 v15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M880 55 h-15 M880 55 v15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M760 205 h15 M760 205 v-15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M880 205 h-15 M880 205 v-15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    {/* Scan line */}
    <line x1="760" y1="130" x2="880" y2="130" stroke="rgba(0,212,255,0.2)" strokeWidth="1" strokeDasharray="4 4" />
  </svg>
);

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isVideo, setIsVideo] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const processFile = (selectedFile) => {
    if (selectedFile && (selectedFile.type.startsWith('image/') || selectedFile.type.startsWith('video/'))) {
      setFile(selectedFile);
      setIsVideo(selectedFile.type.startsWith('video/'));
      setResult(null);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(selectedFile);
    } else {
      setError("Please select a valid image or video file.");
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const analyzeFile = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoint = isVideo ? '/predict/video' : '/predict/image';
      const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Analysis failed. Ensure the backend is running or the file is valid.");
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setIsVideo(false);
  };

  const isFake = result?.prediction === 'Fake';

  return (
    <div className="app-container">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">
            <ShieldIcon />
            <div>
              <h1>SentinelX</h1>
              <span className="header-subtitle">Neural Deepfake Detection Engine</span>
            </div>
          </div>
        </div>
        <div className="header-right">
          <div className="status-pill">
            <span className="status-dot" />
            AI Model Active
          </div>
          <div className="auth-buttons">
            <button className="btn-login">Sign In</button>
            <button className="btn-signup">Get Started</button>
          </div>
        </div>
      </header>

      {/* ── Hero Banner ── */}
      <section className="hero-banner">
        <div className="hero-image-wrapper">
          <HeroBg />
          <div className="hero-overlay">
            <div className="hero-text">
              <h2>Real-Time Deepfake Intelligence</h2>
              <p>Upload any image or video. Our Xception neural network analyzes facial biometrics and flags AI-generated manipulations with high precision.</p>
            </div>
            <div className="hero-stats">
              <div className="stat-box">
                <span className="stat-number">99%</span>
                <span className="stat-label">Accuracy</span>
              </div>
              <div className="stat-box">
                <span className="stat-number">&lt;2s</span>
                <span className="stat-label">Analysis</span>
              </div>
              <div className="stat-box">
                <span className="stat-number">0</span>
                <span className="stat-label">Data Stored</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Main Layout ── */}
      <main className="main-layout">

        {/* Left: Upload Panel */}
        <div className="upload-panel">
          <div className="panel-title">Scan Target</div>

          {!preview ? (
            <div
              className={`drop-zone ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,video/*"
                onChange={handleChange}
                style={{ display: "none" }}
              />
              <div className="upload-icon-wrapper">
                <UploadIcon />
              </div>
              <p>Drop an image or video here</p>
              <span className="file-hint">or click to browse from your device</span>
              <div className="file-types">
                {['JPG', 'PNG', 'WEBP', 'MP4', 'MOV', 'AVI'].map(t => (
                  <span key={t} className="type-badge">{t}</span>
                ))}
              </div>
            </div>
          ) : (
            <div className="preview-container">
              <div className="preview-wrapper">
                {isVideo ? (
                  <video src={preview} controls className="image-preview" />
                ) : (
                  <img src={preview} alt="Upload preview" className="image-preview" />
                )}
                {loading && (
                  <div className="scan-overlay scan-corners">
                    <div className="scan-line" />
                  </div>
                )}
              </div>

              {!result && (
                <button
                  className="analyze-button"
                  onClick={analyzeFile}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" />
                        <path d="M5 8l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Run Deepfake Analysis
                    </>
                  )}
                </button>
              )}

              {result && (
                <button className="reset-button" onClick={resetState}>
                  ↩ Analyze Another File
                </button>
              )}
            </div>
          )}
        </div>

        {/* Right: Info + Result Panel */}
        <div className="info-panel">

          {/* Result */}
          <div className="result-panel">
            <div className="panel-title">Analysis Output</div>

            {!result ? (
              <div className="about-panel" style={{ background: 'transparent', border: 'none', padding: 0 }}>
                <div className="feature-list">
                  {[
                    { icon: '🧠', title: 'Xception Neural Network', desc: 'State-of-the-art deep learning model fine-tuned on 100K+ real and fake face pairs.' },
                    { icon: '🎯', title: 'Face Crop Pipeline', desc: 'Automatically detects and isolates faces for precise deepfake scoring.' },
                    { icon: '🎬', title: 'Video Frame Sampling', desc: 'Samples 15 frames across the video timeline and returns an averaged prediction.' },
                    { icon: '🔒', title: 'Zero Data Retention', desc: 'All processing happens in memory — nothing is stored or logged.' },
                  ].map(f => (
                    <div key={f.title} className="feature-item">
                      <span className="feature-icon">{f.icon}</span>
                      <div>
                        <h4>{f.title}</h4>
                        <p>{f.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className={`result-container ${isFake ? 'fake' : 'real'}`}>
                {/* Verdict */}
                <div className={`result-verdict ${isFake ? 'fake' : 'real'}`}>
                  <span className="verdict-icon">{isFake ? '⚠️' : '✅'}</span>
                  <div className="verdict-text">
                    <div className="label">
                      {isFake
                        ? `Deepfake ${result.analyzed_type === 'video' ? 'Video' : 'Image'} Detected`
                        : `Authentic ${result.analyzed_type === 'video' ? 'Video' : 'Image'}`}
                    </div>
                    <div className="sub">
                      {isFake
                        ? 'High probability of AI-generated manipulation'
                        : 'No significant deepfake artifacts detected'}
                    </div>
                  </div>
                </div>

                {/* Confidence Bar */}
                <div className="confidence-section">
                  <div className="confidence-header">
                    <span className="confidence-label">Confidence Score</span>
                    <span className="confidence-value">{result.confidence.toFixed(1)}%</span>
                  </div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Metadata Grid */}
                <div className="meta-grid">
                  <div className="meta-item">
                    <div className="key">Prediction</div>
                    <div className="value">{result.prediction}</div>
                  </div>
                  <div className="meta-item">
                    <div className="key">Media Type</div>
                    <div className="value" style={{ textTransform: 'capitalize' }}>{result.analyzed_type}</div>
                  </div>
                  <div className="meta-item">
                    <div className="key">Fake Score</div>
                    <div className="value">{result.score_fake?.toFixed(1) ?? '—'}%</div>
                  </div>
                  <div className="meta-item">
                    <div className="key">Real Score</div>
                    <div className="value">{result.score_real?.toFixed(1) ?? '—'}%</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <span>⚠</span>
          {error}
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        SentinelX Lens · Powered by Xception Neural Network · No data stored
      </footer>
    </div>
  );
}

export default App;
