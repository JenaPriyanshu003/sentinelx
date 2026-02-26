import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

// ── Icons ────────────────────────────────────────────────────────────────────
const ShieldIcon = () => (
  <svg className="logo-icon" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '32px', height: '32px' }}>
    <path d="M19 3L5 9v10c0 8.3 5.9 16.1 14 18 8.1-1.9 14-9.7 14-18V9L19 3z"
      stroke="#3b82f6" strokeWidth="2.5" fill="rgba(59,130,246,0.1)" />
    <path d="M19 11v12M15 15l4-4 4 4" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const UploadIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
  </svg>
);

const CloseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);

// ── Auth Modal ───────────────────────────────────────────────────────────────
function AuthModal({ mode, onClose, onSwitch }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    const endpoint = mode === 'login' ? '/auth/login' : '/auth/register';
    const payload = mode === 'login' ? { email, password } : { name, email, password };

    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Auth failed');
      localStorage.setItem('sentinelx_user', JSON.stringify({ name: data.name, email: data.email, token: data.token }));
      window.location.reload();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="drawer-backdrop active" onClick={onClose}>
      <div className="panel auth-panel" onClick={e => e.stopPropagation()} style={{ position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', maxWidth: '450px', width: '90%', zIndex: 2000 }}>
        <div className="panel-title">{mode === 'login' ? 'System Authentication' : 'Establish Identity'}</div>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1rem' }}>
          {mode === 'register' && (
            <input className="btn-auth-outline" style={{ textAlign: 'left', width: '100%', cursor: 'text' }} placeholder="Full Name" type="text" value={name} onChange={e => setName(e.target.value)} required />
          )}
          <input className="btn-auth-outline" style={{ textAlign: 'left', width: '100%', cursor: 'text' }} placeholder="Admin Email" type="email" value={email} onChange={e => setEmail(e.target.value)} required />
          <input className="btn-auth-outline" style={{ textAlign: 'left', width: '100%', cursor: 'text' }} placeholder="Access Crypt-Key" type="password" value={password} onChange={e => setPassword(e.target.value)} required />
          {error && <p style={{ color: 'var(--danger)', fontSize: '13px' }}>{error}</p>}
          <button className="btn-auth-filled" type="submit" style={{ padding: '14px' }}>{mode === 'login' ? 'Authorize Access' : 'Create Credentials'}</button>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'center' }}>
            {mode === 'login' ? "New operative?" : "Already registered?"}
            <span onClick={onSwitch} style={{ color: 'var(--accent-blue)', cursor: 'pointer', marginLeft: '8px', fontWeight: 600 }}>
              {mode === 'login' ? 'Register Identity' : 'Authorized Login'}
            </span>
          </p>
        </form>
      </div>
    </div >
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
  const [user, setUser] = useState(null);
  const [authModal, setAuthModal] = useState(null); // 'login' or 'register'
  const [history, setHistory] = useState([]);
  const [showActivity, setShowActivity] = useState(false);

  const fileInputRef = useRef(null);

  // ── Auth Logic ──
  useEffect(() => {
    const saved = localStorage.getItem('sentinelx_user');
    if (saved) {
      const u = JSON.parse(saved);
      setUser(u);
      fetchHistory(u.email);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('sentinelx_user');
    setUser(null);
    setHistory([]);
    window.location.reload();
  };

  const fetchHistory = async (email) => {
    try {
      const res = await fetch(`http://localhost:8000/history?user_email=${email}`);
      const data = await res.json();
      if (res.ok) setHistory(data);
    } catch (err) { console.error("History fetch error", err); }
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  }, []);

  const processFile = (selectedFile) => {
    if (selectedFile && (selectedFile.type.startsWith('image/') || selectedFile.type.startsWith('video/'))) {
      if (preview) URL.revokeObjectURL(preview);
      setFile(selectedFile);
      setIsVideo(selectedFile.type.startsWith('video/'));
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    } else {
      setError("Invalid file type. Please provide an image or video.");
    }
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  }, []);

  const analyzeFile = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    if (user) formData.append('user_email', user.email);

    const endpoint = isVideo ? '/predict/video' : '/predict/image';
    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: user ? { 'Authorization': `Bearer ${user.token}` } : {},
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Analysis failed');
      setResult(data);
      if (user) fetchHistory(user.email);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetState = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const isFake = result?.prediction === 'Fake';

  return (
    <div className="app-container">
      {/* ── Auth Modal ── */}
      {authModal && (
        <AuthModal
          mode={authModal}
          onClose={() => setAuthModal(null)}
          onSwitch={() => setAuthModal(authModal === 'login' ? 'register' : 'login')}
        />
      )}

      {/* ── Navigation ── */}
      <nav className="header">
        <div className="header-logo">
          <ShieldIcon />
          <h1>SentinelX</h1>
        </div>
        <div className="header-right">
          {user ? (
            <>
              <div className="nav-status">Secured: {user.name}</div>
              <button className="btn-auth-outline" onClick={() => setShowActivity(true)}>Activity Log</button>
              <button className="btn-auth-outline" onClick={handleLogout}>Logout</button>
            </>
          ) : (
            <div className="auth-buttons" style={{ display: 'flex', gap: '12px' }}>
              <button className="btn-auth-outline" onClick={() => setAuthModal('login')}>Sign In</button>
              <button className="btn-auth-filled" onClick={() => setAuthModal('register')}>Get Started</button>
            </div>
          )}
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero-banner">
        <span className="hero-tagline">Advanced Biometric Protection</span>
        <h2>Real-Time Intelligence for Digital Integrity.</h2>
        <p>
          Deploy our Xception-based neural engine to analyze facial artifacts and detect
          AI-generated manipulations with 98% precision.
        </p>

        <div className="hero-stats">
          <div className="stat-item">
            <span className="stat-val">98%</span>
            <span className="stat-label">Accuracy</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">&lt;2s</span>
            <span className="stat-label">Latency</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">256-bit</span>
            <span className="stat-label">Security</span>
          </div>
        </div>
      </section>

      {/* ── Security Hub ── */}
      <section className="main-layout" style={{ marginBottom: '2.5rem' }}>
        <div className="panel security-hub" style={{ width: '100%', gridColumn: '1 / span 2' }}>
          <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>SentinelX Security Hub</span>
            <span style={{ fontSize: '10px', opacity: 0.6 }}>Engine Online: v4.28</span>
          </div>
          <div className="hub-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem', textAlign: 'center', marginTop: '1rem' }}>
            <div className="hub-item">
              <div className="stat-val" style={{ color: 'var(--text-primary)', fontSize: '1.5rem' }}>{history.length}</div>
              <div className="stat-label">Total Assets Analyzed</div>
            </div>
            <div className="hub-item">
              <div className="stat-val" style={{ color: 'var(--text-primary)', fontSize: '1.5rem' }}>{history.filter(i => i.prediction === 'Fake').length}</div>
              <div className="stat-label">Deepfakes Neutralized</div>
            </div>
            <div className="hub-item">
              <div className="stat-val" style={{ color: 'var(--text-primary)', fontSize: '1.5rem' }}>98.2%</div>
              <div className="stat-label">Detection Precision</div>
            </div>
            <div className="hub-item">
              <div className="stat-val" style={{ color: 'var(--text-primary)', fontSize: '1.5rem' }}>&lt;42ms</div>
              <div className="stat-label">Neural Latency</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Main Operations ── */}
      <main className="main-layout">

        {/* Left: Inference Panel */}
        <section className="panel">
          <div className="panel-title">Inference Engine</div>

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

              <div className="upload-icon-box">
                <UploadIcon />
              </div>
              <h3>Analyze Media</h3>
              <p>Drag files here or click to browse</p>

              <div className="file-types" style={{ marginTop: '2rem' }}>
                {['MP4', 'MOV', 'JPG', 'PNG'].map(t => (
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
                {loading && <div className="scan-overlay"><div className="scan-line" /></div>}
              </div>

              {!result ? (
                <button className="btn-primary" onClick={analyzeFile} disabled={loading}>
                  {loading ? 'Processing Stream...' : 'Execute Analysis'}
                </button>
              ) : (
                <button className="btn-auth-outline" style={{ width: '100%', marginTop: '1rem', padding: '12px' }} onClick={resetState}>
                  Discard Result
                </button>
              )}
            </div>
          )}
        </section>

        {/* Right: Info / Results */}
        <section className="panel">
          <div className="panel-title">{!result ? (user ? 'Activity Statistics' : 'Core Capabilities') : 'Analysis Report'}</div>

          {!result ? (
            user ? (
              <div className="history-list">
                <div style={{ padding: '20px', textAlign: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: '15px' }}>
                  <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--accent-blue)' }}>{history.length}</div>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', marginTop: '4px' }}>Total Scans Performed</div>
                </div>
                {history.slice(0, 3).map(item => (
                  <div key={item._id} className="history-item" style={{ marginBottom: '8px', padding: '10px' }}>
                    <div style={{ fontSize: '12px', flex: 1 }}>
                      <strong>{item.prediction}</strong> {item.analyzed_type}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{new Date(item.timestamp).toLocaleDateString()}</div>
                  </div>
                ))}
                <button className="btn-auth-outline" style={{ width: '100%' }} onClick={() => setShowActivity(true)}>View All Records</button>
              </div>
            ) : (
              <div className="feature-list">
                {[
                  { icon: '🧠', title: 'Xception Architecture', desc: 'Custom fine-tuned weights for face anti-spoofing.' },
                  { icon: '🎯', title: 'Biometric Cropping', desc: 'Auto-isolation of high-value facial regions.' },
                  { icon: '🎬', title: 'Temporal Smoothing', desc: 'Frame-by-frame consistency validation.' },
                  { icon: '🔒', title: 'Secure Vault', desc: 'Encrypted storage for sensitive scan records.' }
                ].map(f => (
                  <div key={f.title} className="feature-item">
                    <span className="feature-icon">{f.icon}</span>
                    <div className="item-info">
                      <h4>{f.title}</h4>
                      <p>{f.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : (
            <div className={`result-card ${isFake ? 'fake' : 'real'}`}>
              <div className={`verdict-header ${isFake ? 'fake' : 'real'}`}>
                <div className="verdict-title">
                  {isFake ? 'Deepfake Detected' : 'Authentic Media'}
                </div>
              </div>

              <div className="confidence-scale">
                <div className="scale-label">
                  <span>Confidence Level</span>
                  <span>{result.confidence.toFixed(1)}%</span>
                </div>
                <div className="progress-bar">
                  <div className={`progress-fill ${isFake ? 'fake' : 'real'}`} style={{ width: `${result.confidence}%` }} />
                </div>
              </div>

              <div className="meta-grid">
                <div className="meta-item"><div className="meta-key" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Analyzed</div><div className="meta-val" style={{ fontWeight: 700 }}>{result.analyzed_type}</div></div>
                <div className="meta-item"><div className="meta-key" style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Confidence</div><div className="meta-val" style={{ fontWeight: 700 }}>{result.confidence}%</div></div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* ── Activity Drawer ── */}
      <div className={`drawer-backdrop ${showActivity ? 'active' : ''}`} onClick={() => setShowActivity(false)}>
        <div className="drawer-content" onClick={e => e.stopPropagation()}>
          <div className="drawer-header">
            <h2>Activity History</h2>
            <button className="drawer-close" onClick={() => setShowActivity(false)}><CloseIcon /></button>
          </div>
          <div className="drawer-body">
            {history.length === 0 ? (
              <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '2rem' }}>No activity records found.</p>
            ) : (
              history.map(item => (
                <div key={item._id} className="history-item">
                  <div className={`item-indicator ${item.prediction === 'Fake' ? 'fake' : 'real'}`} style={{ width: '10px', height: '10px', borderRadius: '50%' }} />
                  <div className="item-info">
                    <h4 style={{ color: '#fff' }}>{item.prediction} {item.analyzed_type}</h4>
                    <p style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{new Date(item.timestamp).toLocaleString()}</p>
                  </div>
                  <div className="item-score" style={{ marginLeft: 'auto', fontWeight: 800, color: item.prediction === 'Fake' ? 'var(--danger)' : 'var(--success)' }}>
                    {item.confidence.toFixed(1)}%
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <footer className="footer">
        <div>© 2026 SentinelX Systems · Intelligence for the Digital Era.</div>
        <div style={{ display: 'flex', gap: '20px' }}>
          <span>Terms</span>
          <span>Privacy</span>
          <span>Security API</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
