import React, { useState, useRef, useEffect } from 'react';
import { Columns, SplitSquareVertical, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

export default function ComparisonViewer({ results }) {
  const [viewMode, setViewMode] = useState('slider'); // 'slider' | 'side-by-side' | 'tabs'
  const [sliderPosition, setSliderPosition] = useState(50); // percentage
  const [isDragging, setIsDragging] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeTab, setActiveTab] = useState('enhanced'); // for tabs mode
  const [baselineCompare, setBaselineCompare] = useState('naive'); // 'naive' | 'traditional' | 'original'

  const containerRef = useRef(null);

  const handleMouseDown = () => setIsDragging(true);
  const handleMouseUp = () => setIsDragging(false);

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPosition((x / rect.width) * 100);
  };

  const handleTouchMove = (e) => {
    if (!containerRef.current || !e.touches[0]) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
    setSliderPosition((x / rect.width) * 100);
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => setIsDragging(false);
    window.addEventListener('mouseup', handleGlobalMouseUp);
    return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
  }, []);

  if (!results) return null;

  const { original_b64, naive_b64, traditional_b64, enhanced_b64 } = results;

  const getBeforeImage = () => {
    if (baselineCompare === 'traditional' && traditional_b64) return traditional_b64;
    if (baselineCompare === 'original') return original_b64;
    return naive_b64;
  };

  const getBeforeLabel = () => {
    if (baselineCompare === 'traditional') return 'Traditional ISP (NLM)';
    if (baselineCompare === 'original') return 'Original Image';
    return 'Naive Front-End (Noisy)';
  };

  return (
    <div className="space-y-4">
      {/* Viewer Header / Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-2xl bg-slate-900/60 border border-slate-700/80">
        {/* View Mode Switcher */}
        <div className="flex items-center space-x-1 p-1 rounded-xl bg-slate-800/80 border border-slate-700/60 text-xs font-medium">
          <button
            onClick={() => setViewMode('slider')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition ${
              viewMode === 'slider' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            <span>Split Slider</span>
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition ${
              viewMode === 'side-by-side' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>Side-by-Side</span>
          </button>
        </div>

        {/* Comparison Selector for Slider */}
        {viewMode === 'slider' && (
          <div className="flex items-center space-x-2 text-xs">
            <span className="text-slate-400">Compare with:</span>
            <select
              value={baselineCompare}
              onChange={(e) => setBaselineCompare(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-white rounded-lg px-2.5 py-1 text-xs focus:ring-1 focus:ring-indigo-500"
            >
              <option value="naive">Naive Noisy Front-End</option>
              {traditional_b64 && <option value="traditional">Traditional OpenCV ISP</option>}
              <option value="original">Original Ground Truth</option>
            </select>
          </div>
        )}

        {/* Zoom Controls */}
        <div className="flex items-center space-x-1.5 text-xs text-slate-400">
          <button
            onClick={() => setZoomLevel((z) => Math.max(1, z - 0.5))}
            disabled={zoomLevel <= 1}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="font-mono w-12 text-center text-slate-300">{zoomLevel}x</span>
          <button
            onClick={() => setZoomLevel((z) => Math.min(3, z + 0.5))}
            disabled={zoomLevel >= 3}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-30"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          {zoomLevel > 1 && (
            <button
              onClick={() => setZoomLevel(1)}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 ml-1"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport */}
      {viewMode === 'slider' ? (
        <div
          ref={containerRef}
          onMouseMove={handleMouseMove}
          onTouchMove={handleTouchMove}
          className="relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 select-none cursor-ew-resize aspect-video max-h-[640px] flex items-center justify-center"
        >
          {/* Base Layer: Enhanced Image (Right Side) */}
          <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
            <img
              src={enhanced_b64}
              alt="Enhanced"
              style={{ transform: `scale(${zoomLevel})` }}
              className="w-full h-full object-contain pointer-events-none transition-transform duration-100"
            />
          </div>

          {/* Top Layer: Baseline Image (Left Side clipped) */}
          <div
            className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none"
            style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
          >
            <img
              src={getBeforeImage()}
              alt="Before"
              style={{ transform: `scale(${zoomLevel})` }}
              className="w-full h-full object-contain transition-transform duration-100"
            />
          </div>

          {/* Draggable Divider Line */}
          <div
            onMouseDown={handleMouseDown}
            onTouchStart={() => setIsDragging(true)}
            style={{ left: `${sliderPosition}%` }}
            className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize z-20 shadow-[0_0_12px_rgba(0,0,0,0.8)]"
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-indigo-600 text-white flex items-center justify-center shadow-lg border-2 border-white pointer-events-auto">
              <div className="flex items-center space-x-0.5">
                <span className="w-0.5 h-3 bg-white/70 rounded-full"></span>
                <span className="w-0.5 h-3 bg-white/70 rounded-full"></span>
              </div>
            </div>
          </div>

          {/* Left / Right Floating Tags */}
          <div className="absolute bottom-4 left-4 z-10 px-3 py-1 rounded-lg bg-black/70 backdrop-blur-md border border-white/10 text-xs font-semibold text-amber-300">
            {getBeforeLabel()}
          </div>
          <div className="absolute bottom-4 right-4 z-10 px-3 py-1 rounded-lg bg-indigo-900/80 backdrop-blur-md border border-indigo-400/30 text-xs font-semibold text-indigo-200">
            NEXUS-ISP Enhanced (Hybrid)
          </div>
        </div>
      ) : (
        /* Side-by-Side Comparison Grid */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 1. Naive Noisy Front-End */}
          <div className="rounded-2xl overflow-hidden bg-slate-900/50 border border-slate-800 p-3 flex flex-col">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-xs">
              <span className="font-semibold text-amber-400">1. Naive Front-End</span>
              <span className="text-slate-400 font-mono">No Denoising</span>
            </div>
            <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
              <img
                src={naive_b64}
                alt="Naive Front-End"
                style={{ transform: `scale(${zoomLevel})` }}
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* 2. Hybrid Enhanced */}
          <div className="rounded-2xl overflow-hidden bg-slate-900/50 border border-indigo-500/40 p-3 flex flex-col ring-1 ring-indigo-500/20">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-xs">
              <span className="font-semibold text-indigo-400">2. NEXUS-ISP Hybrid</span>
              <span className="text-emerald-400 font-mono text-[11px] font-semibold">Trained Residual U-Net</span>
            </div>
            <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
              <img
                src={enhanced_b64}
                alt="Enhanced Output"
                style={{ transform: `scale(${zoomLevel})` }}
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* 3. Traditional Classical ISP */}
          {traditional_b64 && (
            <div className="rounded-2xl overflow-hidden bg-slate-900/50 border border-slate-800 p-3 flex flex-col">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-xs">
                <span className="font-semibold text-cyan-400">3. Traditional Classical ISP</span>
                <span className="text-slate-400 font-mono">OpenCV NLM + Unsharp</span>
              </div>
              <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
                <img
                  src={traditional_b64}
                  alt="Traditional ISP"
                  style={{ transform: `scale(${zoomLevel})` }}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}

          {/* 4. Original Ground Truth */}
          <div className="rounded-2xl overflow-hidden bg-slate-900/50 border border-slate-800 p-3 flex flex-col">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-xs">
              <span className="font-semibold text-slate-300">4. Original Image</span>
              <span className="text-slate-500 font-mono">Pre-Simulation Reference</span>
            </div>
            <div className="relative aspect-video rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center">
              <img
                src={original_b64}
                alt="Original Reference"
                style={{ transform: `scale(${zoomLevel})` }}
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
