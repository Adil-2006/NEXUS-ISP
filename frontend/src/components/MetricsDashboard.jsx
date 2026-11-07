import React from 'react';
import { Clock, Zap, TrendingUp, Cpu, Gauge, CheckCircle2 } from 'lucide-react';

export default function MetricsDashboard({ metrics, explanation }) {
  if (!metrics) return null;

  const {
    classical_latency_ms,
    neural_latency_ms,
    total_latency_ms,
    estimated_sigma,
    noise_preset,
    noise_a,
    psnr_hybrid_vs_groundtruth,
    ssim_hybrid_vs_groundtruth,
    psnr_naive_vs_groundtruth,
    ssim_naive_vs_groundtruth,
    psnr_traditional_vs_groundtruth,
    image_shape
  } = metrics;

  const psnrGain = (psnr_hybrid_vs_groundtruth - psnr_naive_vs_groundtruth).toFixed(2);
  const ssimGain = ((ssim_hybrid_vs_groundtruth - ssim_naive_vs_groundtruth) * 100).toFixed(1);

  return (
    <div className="space-y-4">
      {/* Metrics Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {/* Latency Card */}
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider">Total Latency</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-white">
            {total_latency_ms} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 font-mono space-y-0.5">
            <div className="flex justify-between">
              <span>Classical Front:</span>
              <span className="text-slate-300">{classical_latency_ms} ms</span>
            </div>
            <div className="flex justify-between">
              <span>Neural U-Net:</span>
              <span className="text-indigo-400 font-semibold">{neural_latency_ms} ms</span>
            </div>
          </div>
        </div>

        {/* PSNR Card */}
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider">Hybrid PSNR</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">
            {psnr_hybrid_vs_groundtruth}{' '}
            <span className="text-xs font-normal text-slate-400">dB</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 font-mono space-y-0.5">
            <div className="flex justify-between">
              <span>Naive Baseline:</span>
              <span className="text-slate-400">{psnr_naive_vs_groundtruth} dB</span>
            </div>
            <div className="flex justify-between text-emerald-400 font-semibold">
              <span>PSNR Gain:</span>
              <span>+{psnrGain} dB</span>
            </div>
          </div>
        </div>

        {/* SSIM Card */}
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider">Hybrid SSIM</span>
            <Zap className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-indigo-400">
            {ssim_hybrid_vs_groundtruth}
          </div>
          <div className="mt-2 text-[11px] text-slate-400 font-mono space-y-0.5">
            <div className="flex justify-between">
              <span>Naive SSIM:</span>
              <span className="text-slate-400">{ssim_naive_vs_groundtruth}</span>
            </div>
            <div className="flex justify-between text-indigo-400 font-semibold">
              <span>Structural Gain:</span>
              <span>+{ssimGain}%</span>
            </div>
          </div>
        </div>

        {/* Noise & Resolution Card */}
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span className="font-semibold uppercase tracking-wider">Estimated Noise &sigma;</span>
            <Gauge className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-300">
            {estimated_sigma}
          </div>
          <div className="mt-2 text-[11px] text-slate-400 font-mono space-y-0.5">
            <div className="flex justify-between">
              <span>Preset (noise a):</span>
              <span className="text-slate-300">{noise_a}</span>
            </div>
            <div className="flex justify-between">
              <span>Resolution:</span>
              <span className="text-slate-300">{image_shape?.[1]} &times; {image_shape?.[0]}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation Banner */}
      {explanation && (
        <div className="p-3.5 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-300 flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <p className="leading-relaxed">{explanation}</p>
        </div>
      )}
    </div>
  );
}
