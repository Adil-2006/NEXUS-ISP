import React from 'react';
import { X, Layers, Cpu, Sliders, ArrowRight, CheckCircle2, ShieldAlert } from 'lucide-react';

export default function ArchitectureModal({ isOpen, onClose, modelInfo }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-6 sm:p-8">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">NEXUS-ISP Hybrid Architecture</h2>
              <p className="text-xs text-slate-400">Classical Preprocessing & Deep Learning Synergies</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="mt-6 space-y-6 text-sm text-slate-300">
          {/* Highlight Callout: Synthetic RAW Handling */}
          <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-800/50 flex gap-3">
            <ShieldAlert className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-white font-semibold text-xs tracking-wide uppercase">
                Sensor RAW Domain & Synthetic Unprocessing
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                The model is designed specifically for <strong>RAW-to-RGB Image Signal Processing</strong>. Because consumer phones output compressed sRGB, the pipeline applies Brooks et al. (CVPR 2019) <em>unprocessing</em> to invert the image into linear sensor Bayer RGGB RAW with heteroscedastic Poisson-Gaussian sensor noise:
              </p>
              <div className="font-mono text-[11px] bg-slate-950/70 p-2 rounded-lg text-cyan-300 border border-slate-800 mt-2">
                sRGB &rarr; Inv Tone Curve &rarr; Linear sRGB &rarr; Inv CCM &rarr; Inv WB &rarr; Mosaic RGGB + Noise(&sigma;&sup2; = a&middot;x + b)
              </div>
            </div>
          </div>

          {/* Pipeline Stages */}
          <div>
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Two-Stage Hybrid Processing
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                <div className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">Stage 1: Classical Front-End</div>
                <h4 className="text-white font-medium text-sm mb-2">Deterministic Color Science</h4>
                <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
                  <li><strong>Gray-World WB:</strong> Balances illuminant gains (R, G, B)</li>
                  <li><strong>Bayer Demosaicing:</strong> RGGB mosaic interpolation</li>
                  <li><strong>Color Correction Matrix:</strong> Camera to sRGB spectral alignment</li>
                  <li><strong>Tone & Gamma:</strong> Smoothstep S-curve for dynamic range</li>
                  <li><strong>Blind Noise Estimator:</strong> Median Absolute Deviation (MAD) of G1-G2</li>
                </ul>
              </div>

              <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/60">
                <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">Stage 2: Neural Residual U-Net</div>
                <h4 className="text-white font-medium text-sm mb-2">Compact Deep Learning ({modelInfo?.params_k || '63.3'}K Params)</h4>
                <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
                  <li><strong>Inputs:</strong> 4 channels (Classical Front RGB + &sigma; noise map)</li>
                  <li><strong>Depthwise-Separable Convs:</strong> ~8x fewer MACs for real-time mobile speed</li>
                  <li><strong>Residual Formulation:</strong> Learns only &Delta; correction: Output = Front + Net</li>
                  <li><strong>Model Size:</strong> Only ~0.24 MB in FP32; &lt;0.12 MB in FP16</li>
                  <li><strong>Inference:</strong> ~10-15 ms CPU latency, &lt;3 ms on modern GPUs</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Model Specification Card */}
          {modelInfo && (
            <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/60">
              <h3 className="text-white font-semibold text-xs tracking-wider uppercase mb-3 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                Checkpoint Details (Preserved Trained Model)
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div>
                  <span className="text-slate-400 block">Architecture</span>
                  <span className="font-mono font-medium text-white">{modelInfo.mode.toUpperCase()} U-Net</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Parameters</span>
                  <span className="font-mono font-medium text-cyan-400">{modelInfo.params.toLocaleString()} ({modelInfo.params_k}K)</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Base Channels / Depth</span>
                  <span className="font-mono font-medium text-white">base={modelInfo.base}, depth={modelInfo.depth}</span>
                </div>
                <div>
                  <span className="text-slate-400 block">Active Device</span>
                  <span className="font-mono font-medium text-emerald-400">{modelInfo.device?.toUpperCase()}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}
