import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

// ── Icons ────────────────────────────────────────────────────────────────────
const ShieldIcon = () => (
  <svg className="logo-icon" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M19 3L5 9v10c0 8.3 5.9 16.1 14 18 8.1-1.9 14-9.7 14-18V9L19 3z"
      stroke="#00d4ff" strokeWidth="2" fill="rgba(0,212,255,0.08)" />
    <path d="M13 19l4 4 8-8" stroke="#00d4ff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const UploadIcon = () => (
  <svg className="upload-icon-svg" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M16 4v18M8 12l8-8 8 8" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M6 24h20" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
  </svg>
);

const HeroBg = () => (
  <svg className="hero-svg-bg" viewBox="0 0 1100 260" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
    <rect width="1100" height="260" fill="#050b18" />
    {Array.from({ length: 20 }).map((_, i) => (
      <line key={`v${i}`} x1={i * 60} y1="0" x2={i * 60} y2="260" stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
    ))}
    {Array.from({ length: 6 }).map((_, i) => (
      <line key={`h${i}`} x1="0" y1={i * 52} x2="1100" y2={i * 52} stroke="rgba(0,212,255,0.05)" strokeWidth="1" />
    ))}
    <circle cx="750" cy="130" r="180" fill="rgba(0,212,255,0.04)" />
    <circle cx="850" cy="80" r="80" fill="rgba(139,92,246,0.05)" />
    <ellipse cx="820" cy="130" rx="60" ry="75" stroke="rgba(0,212,255,0.15)" strokeWidth="1.5" fill="none" />
    <circle cx="800" cy="115" r="5" fill="rgba(0,212,255,0.3)" />
    <circle cx="840" cy="115" r="5" fill="rgba(0,212,255,0.3)" />
    <path d="M805 148 Q820 158 835 148" stroke="rgba(0,212,255,0.3)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
    <path d="M760 55 h15 M760 55 v15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M880 55 h-15 M880 55 v15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M760 205 h15 M760 205 v-15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <path d="M880 205 h-15 M880 205 v-15" stroke="rgba(0,212,255,0.4)" strokeWidth="2" strokeLinecap="round" />
    <line x1="760" y1="130" x2="880" y2="130" stroke="rgba(0,212,255,0.2)" strokeWidth="1" strokeDasharray="4 4" />
  </svg>
);

// ── Result Modal ─────────────────────────────────────────────────────────────
function ResultModal({ result, onClose }) {
  if (!result) return null;
  const isFake = result.prediction === 'Fake';

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className={`modal-card glass result-modal ${isFake ? 'fake-border' : 'real-border'}`} onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>

        <div className="modal-header">
          <div className={`verdict-badge ${isFake ? 'fake' : 'real'}`}>
            {isFake ? '⚠️ Deepfake Detected' : '✅ Authentic Media'}
          </div>
        </div>

        <div className="result-content">
          <div className="confidence-display">
            <span className="conf-label">Confidence Score</span>
            <div className={`conf-value ${isFake ? 'text-fake' : 'text-real'}`}>{result.confidence.toFixed(1)}%</div>
          </div>

          <div className="conf-bar-tray">
            <div className={`conf-bar-fill ${isFake ? 'fill-fake' : 'fill-real'}`} style={{ width: `${result.confidence}%` }} />
          </div>

          <div className="result-meta-grid">
            <div className="meta-box"><label>Media Type</label><span>{result.analyzed_type}</span></div>
            <div className="meta-box"><label>Timestamp</label><span>{new Date(result.timestamp || Date.now()).toLocaleTimeString()}</span></div>
            <div className="meta-box"><label>Fake Score</label><span>{result.score_fake?.toFixed(1)}%</span></div>
            <div className="meta-box"><label>Real Score</label><span>{result.score_real?.toFixed(1)}%</span></div>
          </div>

          <div className="result-guidance">
            {isFake
              ? "Significant AI-generated artifacts were found. This media is likely manipulated."
              : "No significant deepfake patterns detected. The face appears biologically consistent."}
          </div>
        </div>

        <button className="btn-signup" style={{ marginTop: '1.5rem', width: '100%' }} onClick={onClose}>Dismiss Report</button>
      </div>
    </div>
  );
}

// ── Auth Modal ───────────────────────────────────────────────────────────────
function AuthModal({ mode, onClose, onSwitch, onLogin }) {
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.email || !form.password) {
      setError('Please fill in all required fields.');
      return;
    }
    if (mode === 'signup' && !form.name) {
      setError('Please enter your name.');
      return;
    }
    setLoading(true);
    try {
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/register';
      const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mode === 'login' ? { email: form.email, password: form.password } : form)
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Authentication failed');
      }
      const data = await response.json();
      onLogin(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>

        <div className="modal-header">
          <ShieldIcon />
          <h2 className="modal-title">{mode === 'login' ? 'Welcome Back' : 'Create Account'}</h2>
          <p className="modal-subtitle">
            {mode === 'login'
              ? 'Sign in to access SentinelX Lens'
              : 'Join to start detecting deepfakes'}
          </p>
        </div>

        <form className="modal-form" onSubmit={handleSubmit}>
          {mode === 'signup' && (
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input
                className="form-input"
                type="text"
                placeholder="John Doe"
                value={form.name}
                onChange={e => setForm({ ...form, name: e.target.value })}
              />
            </div>
          )}
          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              className="form-input"
              type="email"
              placeholder="you@email.com"
              value={form.email}
              onChange={e => setForm({ ...form, email: e.target.value })}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              className="form-input"
              type="password"
              placeholder="••••••••"
              value={form.password}
              onChange={e => setForm({ ...form, password: e.target.value })}
            />
          </div>

          {error && <div className="modal-error">⚠ {error}</div>}

          <button className="modal-submit" type="submit" disabled={loading}>
            {loading ? <><div className="spinner" /> Processing...</> : (mode === 'login' ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <p className="modal-switch">
          {mode === 'login' ? "Don't have an account? " : "Already have an account? "}
          <button className="modal-switch-btn" onClick={onSwitch}>
            {mode === 'login' ? 'Sign Up' : 'Sign In'}
          </button>
        </p>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isVideo, setIsVideo] = useState(false);
  const fileInputRef = useRef(null);

  // Auth & DB state
  const [authModal, setAuthModal] = useState(null); // null | 'login' | 'signup'
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('sentinelx_user');
    return saved ? JSON.parse(saved) : null;
  });
  const [history, setHistory] = useState([]);

  const handleLogin = (u) => {
    localStorage.setItem('sentinelx_user', JSON.stringify(u));
    setUser(u);
    setAuthModal(null);
  };

  const handleLogout = () => {
    localStorage.removeItem('sentinelx_user');
    setUser(null);
    setHistory([]);
    setResult(null);
  };

  const fetchHistory = useCallback(async () => {
    if (!user) return;
    try {
      const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/scans/history`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      }
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  }, [user]);

  // Fetch history on user login
  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
  }, []);

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
      const headers = user ? { 'Authorization': `Bearer ${user.token}` } : {};

      const response = await fetch(`${backendUrl}${endpoint}`, { method: 'POST', body: formData, headers });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult(data);
      if (user) fetchHistory(); // refresh history to include the new scan
    } catch (err) {
      setError("Analysis failed. Ensure the backend is running or the file is valid.");
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null); setPreview(null); setResult(null); setError(null); setIsVideo(false);
  };

  const openHistoryItem = (item) => {
    setResult(item);
  };

  const isFake = result?.prediction === 'Fake';

  return (
    <div className="app-container">

      {/* ── Result Modal ── */}
      {result && (
        <ResultModal
          result={result}
          onClose={() => setResult(null)}
        />
      )}

      {/* ── Auth Modal ── */}
      {authModal && (
        <AuthModal
          mode={authModal}
          onClose={() => setAuthModal(null)}
          onSwitch={() => setAuthModal(authModal === 'login' ? 'signup' : 'login')}
          onLogin={handleLogin}
        />
      )}

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
            {user ? (
              <>
                <span className="btn-login" style={{ cursor: 'default', pointerEvents: 'none' }}>👤 {user.name}</span>
                <button className="btn-signup" onClick={handleLogout}>Sign Out</button>
              </>
            ) : (
              <>
                <button className="btn-login" onClick={() => setAuthModal('login')}>Sign In</button>
                <button className="btn-signup" onClick={() => setAuthModal('signup')}>Get Started</button>
              </>
            )}
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
              <p>Upload any image or video. SentinelX analyzes facial biometrics and flags AI-generated manipulations with high precision.</p>
            </div>
            <div className="hero-stats">
              <div className="stat-box"><span className="stat-number">97.5%</span><span className="stat-label">Val Accuracy</span></div>
              <div className="stat-box"><span className="stat-number">&lt;1.8s</span><span className="stat-label">Analysis Speed</span></div>
              <div className="stat-box"><span className="stat-number">Live</span><span className="stat-label">Atlas Secured</span></div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Main Layout ── */}
      <main className="main-layout">

        {/* Left: Upload Panel */}
        <div className="panel-card upload-panel">
          <div className="panel-title">Scan Target</div>
          {!preview ? (
            <div
              className={`drop-zone ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag} onDragLeave={handleDrag}
              onDragOver={handleDrag} onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input ref={fileInputRef} type="file" accept="image/*,video/*"
                onChange={e => e.target.files?.[0] && processFile(e.target.files[0])}
                style={{ display: "none" }} />
              <div className="upload-icon-wrapper"><UploadIcon /></div>
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
                {isVideo
                  ? <video src={preview} controls className="image-preview" />
                  : <img src={preview} alt="preview" className="image-preview" />}
                {loading && <div className="scan-overlay scan-corners"><div className="scan-line" /></div>}
              </div>
              {!result && (
                <button className="analyze-button" onClick={analyzeFile} disabled={loading}>
                  {loading
                    ? <><div className="spinner" /> Analyzing...</>
                    : <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" />
                        <path d="M5 8l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Run Deepfake Analysis
                    </>}
                </button>
              )}
              {result && <button className="reset-button" onClick={resetState}>Analyze Another File</button>}
            </div>
          )}
        </div>

        <div className="panel-card info-panel">
          <div className="panel-title">Scan History</div>
          <div className="history-list">
            {!user ? (
              <div className="empty-history" style={{ textAlign: "center", padding: "2rem" }}>
                <p style={{ marginBottom: "1rem" }}>Sign in to view and save your scan history securely.</p>
                <button className="btn-login" onClick={() => setAuthModal('login')}>Sign In</button>
              </div>
            ) : history.length === 0 ? (
              <div className="empty-history">No past scans found. Run your first analysis!</div>
            ) : (
              history.map(item => (
                <div key={item._id} className="history-card" onClick={() => openHistoryItem(item)}>
                  <div className="history-status">
                    <span className={`status-indicator-mini ${item.prediction === 'Fake' ? 'fake' : 'real'}`} />
                    <span className="history-label">{item.prediction} {item.analyzed_type}</span>
                  </div>
                  <div className="history-meta">
                    <span className="history-conf">{item.confidence.toFixed(1)}%</span>
                    <span className="history-time">{new Date(item.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </main>

      {error && <div className="error-banner"><span>⚠</span>{error}</div>}
      <footer className="footer">SentinelX Lens · Secure MongoDB Atlas Integration Active</footer>
    </div>
  );
}

export default App;
