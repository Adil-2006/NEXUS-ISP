import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import NoiseControls from './components/NoiseControls';
import ComparisonViewer from './components/ComparisonViewer';
import MetricsDashboard from './components/MetricsDashboard';
import DownloadBar from './components/DownloadBar';
import ArchitectureModal from './components/ArchitectureModal';
import { AlertCircle, Sparkles, SlidersHorizontal, RefreshCw } from 'lucide-react';

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Form State
  const [selectedFile, setSelectedFile] = useState(null);
  const [noiseLevel, setNoiseLevel] = useState('medium');
  const [customNoiseA, setCustomNoiseA] = useState(0.002);
  const [includeTraditional, setIncludeTraditional] = useState(true);

  // Processing & Results State
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Fetch initial health and model information
  useEffect(() => {
    async function checkHealthAndInfo() {
      try {
        const healthRes = await fetch('/api/health');
        if (healthRes.ok) {
          const healthData = await healthRes.json();
          setIsHealthy(healthData.status === 'ok');
        }

        const infoRes = await fetch('/api/model-info');
        if (infoRes.ok) {
          const infoData = await infoRes.json();
          setModelInfo(infoData);
        }
      } catch (err) {
        console.warn('Backend connection error:', err);
        setIsHealthy(false);
      }
    }

    checkHealthAndInfo();
  }, []);

  const handleProcessImage = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('noise_level', noiseLevel);
    if (noiseLevel === 'custom') {
      formData.append('custom_noise_a', customNoiseA.toString());
    }
    formData.append('include_traditional', includeTraditional ? 'true' : 'false');

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown processing error' }));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('ISP Processing Error:', err);
      setError(err.message || 'Failed to process image through the ISP pipeline.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Header
        modelInfo={modelInfo}
        isHealthy={isHealthy}
        onOpenInfo={() => setIsModalOpen(true)}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Error Alert */}
        {error && (
          <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center justify-between">
            <div className="flex items-center space-x-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-rose-400 hover:text-white font-semibold ml-4"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Top Control Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Image Selection & Preview (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                Image Input
              </h2>
              {selectedFile && (
                <button
                  onClick={() => { setSelectedFile(null); setResults(null); }}
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition"
                >
                  <RefreshCw className="w-3 h-3" />
                  Reset
                </button>
              )}
            </div>

            <ImageUploader
              selectedFile={selectedFile}
              onSelectFile={(file) => {
                setSelectedFile(file);
                setResults(null);
                setError(null);
              }}
              onClearFile={() => {
                setSelectedFile(null);
                setResults(null);
              }}
              isProcessing={isProcessing}
            />
          </div>

          {/* Right Column: Noise Controls & Run (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <SlidersHorizontal className="w-4 h-4 text-indigo-400" />
              Sensor Configuration
            </h2>

            <NoiseControls
              noiseLevel={noiseLevel}
              setNoiseLevel={setNoiseLevel}
              customNoiseA={customNoiseA}
              setCustomNoiseA={setCustomNoiseA}
              includeTraditional={includeTraditional}
              setIncludeTraditional={setIncludeTraditional}
              onProcess={handleProcessImage}
              isProcessing={isProcessing}
              disabled={!selectedFile}
            />
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <section className="space-y-6 pt-4 border-t border-slate-800/80 animate-in fade-in duration-300">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  Enhancement Results & Analysis
                </h2>
                <p className="text-xs text-slate-400">
                  Interactive Before & After Comparison &bull; Classical vs Hybrid Neural Reconstruction
                </p>
              </div>
            </div>

            {/* Comparison Split Slider / Grid */}
            <ComparisonViewer results={results} />

            {/* Metrics Dashboard */}
            <MetricsDashboard
              metrics={results.metrics}
              explanation={results.explanation}
            />

            {/* Download and Export Bar */}
            <DownloadBar results={results} />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 mt-12 text-center text-xs text-slate-500">
        <p>
          NEXUS-ISP &bull; Hybrid Classical Front-End ISP + 63.35K-Parameter Residual U-Net &bull; Trained on Synthetic RAW Unprocessing
        </p>
      </footer>

      {/* Architecture Deep Dive Modal */}
      <ArchitectureModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        modelInfo={modelInfo}
      />
    </div>
  );
}
