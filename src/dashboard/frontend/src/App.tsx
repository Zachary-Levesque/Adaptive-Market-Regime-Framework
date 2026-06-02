import { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, Legend
} from 'recharts';
import { 
  Activity, Shield, AlertTriangle, TrendingUp, Clock, CheckCircle, XCircle, Info,
  ArrowUpRight, ArrowDownRight, RefreshCw
} from 'lucide-react';

interface RegimeData {
  current_regime: string;
  probabilities: Record<string, number>;
  duration_days: number;
}

interface Performance {
  date: string;
  portfolio_value: number;
  benchmark_value: number;
  drawdown: number;
}

interface RiskMetrics {
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  win_rate: number;
  annual_return: number;
}

interface ReadinessCheck {
  check: string;
  passed: boolean;
  value: string | number;
  detail: string;
}

const REGIME_COLORS: Record<string, string> = {
  "Bull Trending": "#10b981",
  "Low-Vol Compression": "#3b82f6",
  "Bear Trending": "#f59e0b",
  "High-Vol Crisis": "#ef4444"
};

export default function App() {
  const [regime, setRegime] = useState<RegimeData | null>(null);
  const [performance, setPerformance] = useState<Performance[]>([]);
  const [risk, setRisk] = useState<RiskMetrics | null>(null);
  const [readiness, setReadiness] = useState<ReadinessCheck[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [regRes, perfRes, riskRes, readRes] = await Promise.all([
        fetch('/api/regime/current'),
        fetch('/api/portfolio/performance'),
        fetch('/api/risk/metrics'),
        fetch('/api/readiness')
      ]);

      const [regData, perfData, riskData, readData] = await Promise.all([
        regRes.json(), perfRes.json(), riskRes.json(), readRes.json()
      ]);

      setRegime(regData);
      setPerformance(perfData);
      setRisk(riskData);
      setReadiness(readData);
    } catch (err) {
      console.error("Failed to fetch data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Poll every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
      </div>
    );
  }

  const currentRegimeColor = regime ? REGIME_COLORS[regime.current_regime] || '#64748b' : '#64748b';

  return (
    <div className="min-h-screen p-6 space-y-6">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">AMRF Dashboard</h1>
          <p className="text-slate-400">Adaptive Market Regime Framework • Live Diagnostics</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 bg-slate-900 rounded-full border border-slate-800">
            <Clock className="h-4 w-4 text-slate-500" />
            <span className="text-sm font-medium">Last updated: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </header>

      {/* Top Row: Regime & Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="md:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <Activity className="h-24 w-24" />
          </div>
          <div className="flex items-center gap-3 mb-4">
            <Shield className="h-5 w-5 text-blue-400" />
            <h2 className="text-lg font-semibold">Market Regime</h2>
          </div>
          <div className="flex items-end gap-4 mb-6">
            <div className="text-4xl font-bold" style={{ color: currentRegimeColor }}>
              {regime?.current_regime}
            </div>
            <div className="text-slate-500 mb-1">
              Active for {regime?.duration_days} days
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {regime && Object.entries(regime.probabilities).map(([key, value]) => (
              <div key={key} className="space-y-1">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 truncate">Regime {key}</div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-blue-500 transition-all duration-1000" 
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
                <div className="text-xs font-mono">{(value * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="h-5 w-5 text-emerald-400" />
            <h2 className="text-lg font-semibold">Sharpe Ratio</h2>
          </div>
          <div className="text-4xl font-bold mb-2">
            {risk?.sharpe.toFixed(2)}
          </div>
          <div className="flex items-center gap-1 text-sm">
            {risk && risk.sharpe >= 0.5 ? (
              <span className="text-emerald-400 flex items-center gap-1"><CheckCircle className="h-3 w-3" /> Passing Grade</span>
            ) : (
              <span className="text-red-400 flex items-center gap-1"><AlertTriangle className="h-3 w-3" /> Below Threshold (0.5)</span>
            )}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="h-5 w-5 text-amber-400" />
            <h2 className="text-lg font-semibold">Max Drawdown</h2>
          </div>
          <div className="text-4xl font-bold mb-2 text-red-400">
            {(risk?.max_drawdown ? risk.max_drawdown * 100 : 0).toFixed(1)}%
          </div>
          <div className="text-slate-500 text-sm">
            Calmar Ratio: {risk?.calmar.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg font-semibold">Equity Curve</h2>
            <div className="flex gap-4 text-xs">
              <div className="flex items-center gap-1.5"><div className="h-2 w-2 rounded-full bg-blue-500" /> Strategy</div>
              <div className="flex items-center gap-1.5"><div className="h-2 w-2 rounded-full bg-slate-600" /> SPY Benchmark</div>
            </div>
          </div>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={performance}>
                <defs>
                  <linearGradient id="colorStrategy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickFormatter={(val) => new Date(val).toLocaleDateString()}
                  minTickGap={50}
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  labelStyle={{ color: '#94a3b8' }}
                  itemStyle={{ fontSize: '12px' }}
                />
                <Area type="monotone" dataKey="portfolio_value" stroke="#3b82f6" fillOpacity={1} fill="url(#colorStrategy)" />
                <Line type="monotone" dataKey="benchmark_value" stroke="#475569" strokeDasharray="5 5" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-6">Readiness Gate</h2>
          <div className="space-y-4">
            {readiness.slice(0, 8).map((item, idx) => (
              <div key={idx} className="flex items-start justify-between p-3 bg-slate-950/50 rounded-lg border border-slate-800">
                <div className="space-y-1">
                  <div className="text-xs font-medium text-slate-300">{item.check.replace(/_/g, ' ').toUpperCase()}</div>
                  <div className="text-[10px] text-slate-500">{item.detail}</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono">{typeof item.value === 'number' ? item.value.toFixed(3) : item.value}</span>
                  {item.passed ? (
                    <CheckCircle className="h-4 w-4 text-emerald-500 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500 shrink-0" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
