import React from 'react';
import { Sliders, Sun, Sunset, Moon, Sparkles, Loader2 } from 'lucide-react';

const PRESETS = [
  {
    id: 'low',
    name: 'Low Noise',
    desc: 'Daylight / Bright',
    a: '2.0e-4',
    icon: Sun,
    color: 'text-amber-400'
  },
  {
    id: 'medium',
    name: 'Medium Noise',
    desc: 'Indoor / Typical',
    a: '2.0e-3',
    icon: Sunset,
    color: 'text-orange-400'
  },
  {
    id: 'high',
    name: 'High Noise',
    desc: 'Low Light / High ISO',
    a: '1.0e-2',
    icon: Moon,
    color: 'text-indigo-400'
  }
];

export default function NoiseControls({
  noiseLevel,
  setNoiseLevel,
  customNoiseA,
  setCustomNoiseA,
  includeTraditional,
  setIncludeTraditional,
  onProcess,
  isProcessing,
  disabled
}) {
  return (
    <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/60 border border-slate-700/80 space-y-5">
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-400" />
            Sensor Noise Condition
          </label>
          <span className="text-[11px] text-slate-400 font-mono">
            Poisson-Gaussian parameter <em>a</em>
          </span>
        </div>

        {/* Preset Cards */}
        <div className="grid grid-cols-3 gap-2.5">
          {PRESETS.map((preset) => {
            const Icon = preset.icon;
            const isSelected = noiseLevel === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                onClick={() => setNoiseLevel(preset.id)}
                className={`p-3 rounded-xl border text-left transition-all duration-150 ${
                  isSelected
                    ? 'bg-indigo-600/15 border-indigo-500 shadow-sm shadow-indigo-500/20 ring-1 ring-indigo-500/30'
                    : 'bg-slate-800/40 border-slate-700/60 hover:bg-slate-800/70 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <Icon className={`w-4 h-4 ${preset.color}`} />
                  <span className="text-[10px] font-mono text-slate-400">{preset.a}</span>
                </div>
                <div className={`text-xs font-semibold ${isSelected ? 'text-white' : 'text-slate-300'}`}>
                  {preset.name}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">{preset.desc}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Traditional ISP Comparison Toggle */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-xs">
        <label className="flex items-center space-x-2.5 cursor-pointer text-slate-300">
          <input
            type="checkbox"
            checked={includeTraditional}
            onChange={(e) => setIncludeTraditional(e.target.checked)}
            className="rounded border-slate-700 bg-slate-800 text-indigo-600 focus:ring-indigo-500/30 h-4 w-4"
          />
          <span>Include Classical OpenCV ISP baseline (NLM + Unsharp)</span>
        </label>
      </div>

      {/* Run Action Button */}
      <button
        onClick={onProcess}
        disabled={disabled || isProcessing}
        className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white font-semibold text-sm shadow-lg shadow-indigo-600/25 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-white" />
            <span>Processing Pipeline...</span>
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-indigo-200" />
            <span>Run NEXUS-ISP Enhancement</span>
          </>
        )}
      </button>
    </div>
  );
}
