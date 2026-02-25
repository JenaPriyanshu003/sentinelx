import { useState, useRef, useCallback } from 'react';
import './App.css';

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
      reader.onloadend = () => {
        setPreview(reader.result);
      };
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

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoint = isVideo ? '/predict/video' : '/predict/image';
      // Backend should be running on localhost:8000
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to analyze file. No face detected, or the backend is offline.");
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

  return (
    <div className="app-container">
      <header className="header">
        <h1>SentinelX Lens</h1>
        <p>Advanced AI Deepfake Detection</p>
      </header>

      <main className="upload-card">
        {!preview ? (
          <div
            className={`drop-zone ${dragActive ? "drag-active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              onChange={handleChange}
              style={{ display: "none" }}
            />
            <span className="upload-icon">📸 / 🎥</span>
            <p>Drag and drop an image or video here</p>
            <span className="file-hint">or click to browse from your device</span>
          </div>
        ) : (
          <div className="preview-container">
            {isVideo ? (
              <video src={preview} controls className="image-preview" />
            ) : (
              <img src={preview} alt="Upload preview" className="image-preview" />
            )}

            {!result ? (
              <button
                className="analyze-button"
                onClick={analyzeImage}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  "Analyze for Deepfakes"
                )}
              </button>
            ) : (
              <div className={`result-container ${result.prediction.toLowerCase()}`}>
                <div className="result-badge">
                  {result.prediction === "Fake"
                    ? `⚠ Deepfake ${result.analyzed_type === 'video' ? 'Video' : 'Image'} Detected`
                    : `✓ Authentic ${result.analyzed_type === 'video' ? 'Video' : 'Image'}`}
                </div>

                <div className="confidence-meter">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence}%` }}
                  ></div>
                </div>
                <span className="confidence-text">
                  {result.confidence.toFixed(1)}% Confidence
                </span>

                <button className="reset-button" onClick={resetState}>
                  Check Another File
                </button>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
