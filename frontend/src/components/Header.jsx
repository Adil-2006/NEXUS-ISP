import React from 'react';
import { Cpu, Sparkles, Activity, Info, ShieldCheck } from 'lucide-react';

export default function Header({ modelInfo, isHealthy, onOpenInfo }) {
  return (
    <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/25 ring-1 ring-white/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                NEXUS-ISP
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  v1.0 Hybrid
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Classical Front-End ISP + 63K-Param Residual Deep Learning
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {/* Status Badge */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs">
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
            <span className="text-slate-300 font-medium">
              {isHealthy ? 'Backend Ready' : 'Connecting...'}
            </span>
            {modelInfo && (
              <>
                <span className="text-slate-600">|</span>
                <span className="text-cyan-400 font-mono flex items-center gap-1">
                  <Cpu className="w-3 h-3" />
                  {modelInfo.device?.toUpperCase()}
                </span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-400 font-mono">
                  {modelInfo.params_k}K params
                </span>
              </>
            )}
          </div>

          {/* Architecture Details Button */}
          <button
            onClick={onOpenInfo}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700 flex items-center gap-1.5 text-xs font-medium"
            title="View Pipeline Architecture"
          >
            <Info className="w-4 h-4 text-indigo-400" />
            <span className="hidden md:inline">Pipeline Details</span>
          </button>
        </div>
      </div>
    </header>
  );
}
