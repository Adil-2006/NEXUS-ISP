import React, { useState } from 'react';
import { Download, FileImage, Copy, Check } from 'lucide-react';

export default function DownloadBar({ results }) {
  const [copied, setCopied] = useState(false);
  if (!results) return null;

  const { enhanced_b64, naive_b64, traditional_b64, metrics } = results;

  const downloadImage = (dataUrl, filename) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const copyMetrics = () => {
    const text = `NEXUS-ISP Benchmark Results:
- Total Latency: ${metrics.total_latency_ms} ms (Classical: ${metrics.classical_latency_ms} ms, Neural: ${metrics.neural_latency_ms} ms)
- Noise Condition: ${metrics.noise_preset} (a=${metrics.noise_a}, sigma=${metrics.estimated_sigma})
- PSNR: ${metrics.psnr_hybrid_vs_groundtruth} dB (Naive: ${metrics.psnr_naive_vs_groundtruth} dB)
- SSIM: ${metrics.ssim_hybrid_vs_groundtruth} (Naive: ${metrics.ssim_naive_vs_groundtruth})
- Resolution: ${metrics.image_shape?.[1]}x${metrics.image_shape?.[0]}
- Architecture: Hybrid Residual U-Net (63.35K parameters)`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-700/80 flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center space-x-2 text-xs text-slate-300">
        <FileImage className="w-4 h-4 text-indigo-400" />
        <span className="font-semibold text-white">Export Results:</span>
        <span className="text-slate-500 hidden sm:inline">&bull; Full resolution restored frame</span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={copyMetrics}
          className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium flex items-center gap-1.5 transition"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
          <span>{copied ? 'Metrics Copied!' : 'Copy Metrics'}</span>
        </button>

        <button
          onClick={() => downloadImage(enhanced_b64, 'nexus_isp_enhanced.jpg')}
          className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/30 flex items-center gap-2 transition"
        >
          <Download className="w-4 h-4" />
          <span>Download Enhanced Image</span>
        </button>

        <button
          onClick={() => downloadImage(naive_b64, 'nexus_isp_naive_noisy.jpg')}
          className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-medium flex items-center gap-1.5 transition"
          title="Download the noisy classical front-end baseline"
        >
          <Download className="w-3.5 h-3.5 text-slate-400" />
          <span>Download Naive</span>
        </button>
      </div>
    </div>
  );
}
