import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, Sparkles, Check } from 'lucide-react';

// Built-in synthesized sample presets for instant 1-click testing
const SAMPLES = [
  {
    name: 'Geometric City Lights',
    category: 'Night Photography',
    generate: () => createSampleCanvas('night')
  },
  {
    name: 'Vibrant Sunset Landscape',
    category: 'High Dynamic Range',
    generate: () => createSampleCanvas('sunset')
  },
  {
    name: 'Macro Floral Pattern',
    category: 'Fine Texture',
    generate: () => createSampleCanvas('macro')
  }
];

function createSampleCanvas(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 384;
  canvas.height = 384;
  const ctx = canvas.getContext('2d');

  if (type === 'night') {
    // Dark sky with city glow and lights
    const sky = ctx.createLinearGradient(0, 0, 0, 384);
    sky.addColorStop(0, '#020617');
    sky.addColorStop(0.7, '#0f172a');
    sky.addColorStop(1, '#1e1b4b');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 384, 384);

    // City skyline silhouettes
    ctx.fillStyle = '#090d16';
    ctx.fillRect(40, 200, 60, 184);
    ctx.fillRect(110, 150, 80, 234);
    ctx.fillRect(200, 180, 70, 204);
    ctx.fillRect(280, 120, 65, 264);

    // Illuminated windows
    ctx.fillStyle = '#fef08a';
    for (let x = 120; x < 180; x += 15) {
      for (let y = 160; y < 350; y += 20) {
        if (Math.random() > 0.3) ctx.fillRect(x, y, 6, 8);
      }
    }
    // Neon glow
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(235, 230, 16, 0, Math.PI * 2);
    ctx.fill();
  } else if (type === 'sunset') {
    // Rich sunset gradient
    const sun = ctx.createLinearGradient(0, 0, 0, 384);
    sun.addColorStop(0, '#312e81');
    sun.addColorStop(0.3, '#c026d3');
    sun.addColorStop(0.6, '#f97316');
    sun.addColorStop(0.8, '#fde047');
    sun.addColorStop(1, '#064e3b');
    ctx.fillStyle = sun;
    ctx.fillRect(0, 0, 384, 384);

    // Mountain ridges
    ctx.fillStyle = '#1e1b4b';
    ctx.beginPath();
    ctx.moveTo(0, 300);
    ctx.lineTo(100, 240);
    ctx.lineTo(200, 270);
    ctx.lineTo(300, 220);
    ctx.lineTo(384, 280);
    ctx.lineTo(384, 384);
    ctx.lineTo(0, 384);
    ctx.closePath();
    ctx.fill();
  } else {
    // Macro floral spiral
    const bg = ctx.createRadialGradient(192, 192, 20, 192, 192, 200);
    bg.addColorStop(0, '#f43f5e');
    bg.addColorStop(0.5, '#881337');
    bg.addColorStop(1, '#1e293b');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 384, 384);

    ctx.strokeStyle = '#fde047';
    ctx.lineWidth = 3;
    for (let i = 0; i < 40; i++) {
      const angle = 0.3 * i;
      const r = 4 * i;
      const x = 192 + r * Math.cos(angle);
      const y = 192 + r * Math.sin(angle);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      resolve(new File([blob], `${type}_sample.jpg`, { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.95);
  });
}

export default function ImageUploader({ selectedFile, onSelectFile, onClearFile, isProcessing }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onSelectFile(file);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onSelectFile(e.target.files[0]);
    }
  };

  const handleSelectSample = async (sample) => {
    const file = await sample.generate();
    onSelectFile(file);
  };

  return (
    <div className="space-y-4">
      {/* Upload Dropzone */}
      {!selectedFile ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragOver
              ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
              : 'border-slate-700/80 hover:border-slate-600 bg-slate-900/40 hover:bg-slate-900/70'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
          />
          <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
            <UploadCloud className="w-7 h-7" />
          </div>
          <h3 className="text-sm font-semibold text-white mb-1">
            Drop an image here, or <span className="text-indigo-400 underline decoration-indigo-400/40">browse files</span>
          </h3>
          <p className="text-xs text-slate-400">
            Supports PNG, JPEG, WebP (standard sRGB images will be unprocessed to RAW)
          </p>
        </div>
      ) : (
        /* Image Preview Card */
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80 flex items-center justify-between">
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="w-14 h-14 rounded-xl overflow-hidden bg-slate-800 border border-slate-700 shrink-0">
              <img
                src={URL.createObjectURL(selectedFile)}
                alt="Upload preview"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="min-w-0">
              <h4 className="text-xs font-semibold text-white truncate max-w-xs sm:max-w-md">
                {selectedFile.name}
              </h4>
              <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                {(selectedFile.size / 1024).toFixed(1)} KB &bull; {selectedFile.type.replace('image/', '').toUpperCase()}
              </p>
            </div>
          </div>
          <button
            onClick={onClearFile}
            disabled={isProcessing}
            className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition disabled:opacity-50"
            title="Remove image"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Built-in Sample Presets */}
      {!selectedFile && (
        <div className="pt-2">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            Or try a quick test sample
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {SAMPLES.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectSample(sample)}
                className="p-3 rounded-xl bg-slate-900/50 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 text-left transition group"
              >
                <div className="text-xs font-medium text-slate-200 group-hover:text-indigo-300">
                  {sample.name}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  {sample.category}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
