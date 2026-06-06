import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Gauge,
  Layers3,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingUp,
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

interface AllocationPolicyRow {
  regime: string;
  alpha_exposure: number;
  benchmark: string;
  benchmark_exposure: number;
}

interface CurrentAllocation {
  date: string;
  alpha_exposure: number;
  benchmark_exposure: number;
}

const REGIME_COLORS: Record<string, string> = {
  'Bull Trending': '#0f9f6e',
  'Low-Vol Compression': '#2563eb',
  'Bear Trending': '#d97706',
  'High-Vol Crisis': '#dc2626',
};

const REGIME_LABELS: Record<string, string> = {
  '0': 'Bull Trending',
  '1': 'Low-Vol Compression',
  '2': 'Bear Trending',
  '3': 'High-Vol Crisis',
};

export default function App() {
  const [regime, setRegime] = useState<RegimeData | null>(null);
  const [performance, setPerformance] = useState<Performance[]>([]);
  const [risk, setRisk] = useState<RiskMetrics | null>(null);
  const [readiness, setReadiness] = useState<ReadinessCheck[]>([]);
  const [allocationPolicy, setAllocationPolicy] = useState<AllocationPolicyRow[]>([]);
  const [currentAllocation, setCurrentAllocation] = useState<CurrentAllocation | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = async (manual = false) => {
    if (manual) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setError(null);

    try {
      const [regData, perfData, riskData, readData] = await Promise.all([
        fetchJson<RegimeData>('/api/regime/current'),
        fetchJson<Performance[]>('/api/portfolio/performance'),
        fetchJson<RiskMetrics>('/api/risk/metrics'),
        fetchJson<ReadinessCheck[]>('/api/readiness'),
      ]);

      setRegime(regData);
      setPerformance(perfData);
      setRisk(riskData);
      setReadiness(readData);

      const [policyResult, allocationResult] = await Promise.allSettled([
        fetchJson<AllocationPolicyRow[]>('/api/allocation/policy'),
        fetchJson<CurrentAllocation>('/api/allocation/current'),
      ]);

      if (policyResult.status === 'fulfilled') {
        setAllocationPolicy(policyResult.value);
      }
      if (allocationResult.status === 'fulfilled') {
        setCurrentAllocation(allocationResult.value);
      }

      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load dashboard data.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => fetchData(), 60000);
    return () => clearInterval(interval);
  }, []);

  const readinessSummary = useMemo(() => {
    const passed = readiness.filter((item) => item.passed).length;
    const total = readiness.length;
    return {
      passed,
      total,
      ready: total > 0 && passed === total,
    };
  }, [readiness]);

  if (loading) {
    return <LoadingScreen />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => fetchData(true)} />;
  }

  return (
    <main className="min-h-screen bg-[#f6f8fb] text-slate-950">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-5 sm:px-6 lg:px-8 lg:py-8">
        <Header
          lastUpdated={lastUpdated}
          refreshing={refreshing}
          onRefresh={() => fetchData(true)}
        />

        <HeroSummary
          regime={regime}
          risk={risk}
          readiness={readinessSummary}
          currentAllocation={currentAllocation}
        />

        <section className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.55fr)_minmax(360px,0.95fr)]">
          <EquityCurveCard performance={performance} />
          <AllocationCard
            currentAllocation={currentAllocation}
            allocationPolicy={allocationPolicy}
            regime={regime}
          />
        </section>

        <section className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <RegimeCard regime={regime} />
          <ReadinessCard readiness={readiness} />
        </section>
      </div>
    </main>
  );
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed for ${url}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function Header({
  lastUpdated,
  refreshing,
  onRefresh,
}: {
  lastUpdated: Date | null;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  return (
    <header className="flex flex-col gap-4 rounded-[28px] border border-white/80 bg-white/90 p-5 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-500">
          <Sparkles className="h-4 w-4 text-blue-600" aria-hidden="true" />
          Research dashboard
        </div>
        <h1 className="text-balance text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">
          Adaptive Market Regime Framework
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
          Regime state, portfolio allocation, risk quality, and readiness in one view.
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:items-end">
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
          <Clock3 className="h-4 w-4 text-slate-500" aria-hidden="true" />
          <span>{lastUpdated ? `Updated ${formatTime(lastUpdated)}` : 'Awaiting refresh'}</span>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          className="inline-flex items-center justify-center gap-2 rounded-full bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:-translate-y-0.5 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
          aria-label="Refresh dashboard data"
        >
          {refreshing ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
          )}
          Refresh
        </button>
      </div>
    </header>
  );
}

function HeroSummary({
  regime,
  risk,
  readiness,
  currentAllocation,
}: {
  regime: RegimeData | null;
  risk: RiskMetrics | null;
  readiness: { passed: number; total: number; ready: boolean };
  currentAllocation: CurrentAllocation | null;
}) {
  const regimeColor = regime ? REGIME_COLORS[regime.current_regime] || '#2563eb' : '#64748b';

  return (
    <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        icon={Activity}
        label="Current regime"
        value={regime?.current_regime ?? 'Unavailable'}
        helper={regime ? `${regime.duration_days} active trading days` : 'No regime artifact loaded'}
        accent={regimeColor}
      />
      <MetricCard
        icon={TrendingUp}
        label="Strategy Sharpe"
        value={risk ? risk.sharpe.toFixed(2) : 'N/A'}
        helper={risk ? `${formatPercent(risk.annual_return)} annual return` : 'No performance artifact loaded'}
        accent="#0f9f6e"
      />
      <MetricCard
        icon={Gauge}
        label="Max drawdown"
        value={risk ? formatPercent(Math.abs(risk.max_drawdown)) : 'N/A'}
        helper={risk ? `Calmar ${risk.calmar.toFixed(2)}` : 'No risk metrics loaded'}
        accent="#dc2626"
      />
      <MetricCard
        icon={ShieldCheck}
        label="Readiness"
        value={readiness.ready ? 'Ready' : 'Review'}
        helper={readiness.total ? `${readiness.passed}/${readiness.total} checks passing` : 'No readiness checks'}
        accent={readiness.ready ? '#0f9f6e' : '#d97706'}
        badge={currentAllocation ? `${formatPercent(currentAllocation.alpha_exposure)} alpha` : undefined}
      />
    </section>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  helper,
  accent,
  badge,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  helper: string;
  accent: string;
  badge?: string;
}) {
  return (
    <article className="group rounded-3xl border border-white/80 bg-white p-5 shadow-[0_18px_60px_rgba(15,23,42,0.07)] transition hover:-translate-y-0.5 hover:shadow-[0_26px_80px_rgba(15,23,42,0.1)]">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div
          className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-50 ring-1 ring-slate-100"
          style={{ color: accent }}
        >
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
        {badge ? <StatusPill tone="neutral">{badge}</StatusPill> : null}
      </div>
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-1 truncate text-2xl font-semibold tracking-tight text-slate-950">{value}</p>
      <p className="mt-2 text-sm leading-5 text-slate-500">{helper}</p>
    </article>
  );
}

function EquityCurveCard({ performance }: { performance: Performance[] }) {
  return (
    <Card>
      <SectionHeader
        icon={BarChart3}
        title="Equity curve"
        description="Strategy and SPY benchmark over the latest available window."
        action={<LegendItem />}
      />
      {performance.length ? (
        <div className="mt-5 h-[360px] w-full sm:h-[430px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={performance} margin={{ top: 12, right: 18, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="strategyFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#2563eb" stopOpacity={0.24} />
                  <stop offset="100%" stopColor="#2563eb" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 6" vertical={false} />
              <XAxis
                dataKey="date"
                axisLine={false}
                tickLine={false}
                minTickGap={42}
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={formatChartDate}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                width={52}
                tick={{ fill: '#64748b', fontSize: 12 }}
                tickFormatter={(value) => `${Number(value).toFixed(1)}x`}
              />
              <Tooltip
                cursor={{ stroke: '#94a3b8', strokeDasharray: '4 4' }}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e2e8f0',
                  borderRadius: 16,
                  boxShadow: '0 24px 60px rgba(15, 23, 42, 0.14)',
                }}
                labelFormatter={(value) => formatLongDate(String(value))}
                formatter={(value: number, name: string) => [
                  `${Number(value).toFixed(2)}x`,
                  name === 'portfolio_value' ? 'Strategy' : 'SPY',
                ]}
              />
              <Area
                type="monotone"
                dataKey="portfolio_value"
                stroke="#2563eb"
                strokeWidth={2.5}
                fill="url(#strategyFill)"
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="benchmark_value"
                stroke="#64748b"
                strokeWidth={2}
                strokeDasharray="6 6"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyState
          icon={BarChart3}
          title="No performance history"
          description="Run the Phase 4 backtest to generate backtest_results.parquet."
        />
      )}
    </Card>
  );
}

function AllocationCard({
  currentAllocation,
  allocationPolicy,
  regime,
}: {
  currentAllocation: CurrentAllocation | null;
  allocationPolicy: AllocationPolicyRow[];
  regime: RegimeData | null;
}) {
  const alpha = currentAllocation?.alpha_exposure ?? 0;
  const benchmark = currentAllocation?.benchmark_exposure ?? 0;

  return (
    <Card>
      <SectionHeader
        icon={Layers3}
        title="Portfolio allocation"
        description="The final portfolio blends the alpha sleeve with SPY by regime."
      />

      {currentAllocation ? (
        <div className="mt-6">
          <div className="rounded-3xl bg-slate-950 p-5 text-white shadow-inner">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm text-slate-300">Current blend</p>
                <p className="mt-1 text-3xl font-semibold tracking-tight">
                  {formatPercent(alpha)} alpha
                </p>
              </div>
              <StatusPill tone="success">{regime?.current_regime ?? 'Regime aware'}</StatusPill>
            </div>
            <div className="mt-5 h-3 overflow-hidden rounded-full bg-white/10">
              <div
                className="h-full rounded-full bg-blue-400 transition-all duration-500"
                style={{ width: `${alpha * 100}%` }}
                aria-label={`Alpha sleeve ${formatPercent(alpha)}`}
              />
            </div>
            <div className="mt-3 flex justify-between text-sm text-slate-300">
              <span>Alpha {formatPercent(alpha)}</span>
              <span>SPY {formatPercent(benchmark)}</span>
            </div>
          </div>

          {allocationPolicy.length ? (
            <div className="mt-5 space-y-3">
              {allocationPolicy.map((row) => (
                <AllocationRow key={row.regime} row={row} />
              ))}
            </div>
          ) : (
            <EmptyState
              compact
              icon={Layers3}
              title="Policy unavailable"
              description="The current exposure is loaded, but the allocation policy artifact is missing."
            />
          )}
        </div>
      ) : (
        <EmptyState
          icon={Layers3}
          title="No allocation artifact"
          description="Run Phase 4 to write allocation_exposure.parquet and allocation_policy.parquet."
        />
      )}
    </Card>
  );
}

function AllocationRow({ row }: { row: AllocationPolicyRow }) {
  const regimeName = REGIME_LABELS[String(row.regime)] ?? `Regime ${row.regime}`;
  const color = REGIME_COLORS[regimeName] || '#2563eb';

  return (
    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 transition hover:border-slate-300 hover:bg-white">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-900">{regimeName}</p>
          <p className="text-xs text-slate-500">Regime {row.regime}</p>
        </div>
        <span className="text-sm font-semibold text-slate-700">
          {formatPercent(row.alpha_exposure)} alpha
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full"
          style={{ width: `${row.alpha_exposure * 100}%`, backgroundColor: color }}
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        {formatPercent(row.benchmark_exposure)} {row.benchmark}
      </p>
    </div>
  );
}

function RegimeCard({ regime }: { regime: RegimeData | null }) {
  return (
    <Card>
      <SectionHeader
        icon={Activity}
        title="Regime probabilities"
        description="Current HMM probability mix across market states."
      />
      {regime && Object.keys(regime.probabilities).length ? (
        <div className="mt-5 space-y-4">
          {Object.entries(regime.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([name, probability]) => (
              <div key={name}>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="text-sm font-medium text-slate-800">{name}</span>
                  <span className="font-mono text-sm text-slate-500">{formatPercent(probability)}</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${probability * 100}%`,
                      backgroundColor: REGIME_COLORS[name] || '#2563eb',
                    }}
                  />
                </div>
              </div>
            ))}
        </div>
      ) : (
        <EmptyState
          icon={Activity}
          title="No regime probabilities"
          description="Regime probability artifacts are not available yet."
        />
      )}
    </Card>
  );
}

function ReadinessCard({ readiness }: { readiness: ReadinessCheck[] }) {
  const failures = readiness.filter((item) => !item.passed);
  const visible = failures.length ? failures : readiness.slice(0, 8);

  return (
    <Card>
      <SectionHeader
        icon={ShieldCheck}
        title="Readiness gate"
        description={
          failures.length
            ? 'Checks that need attention before deployment.'
            : 'All critical readiness checks are currently passing.'
        }
        action={
          <StatusPill tone={failures.length ? 'warning' : 'success'}>
            {failures.length ? `${failures.length} open` : 'All clear'}
          </StatusPill>
        }
      />

      {readiness.length ? (
        <div className="mt-5 divide-y divide-slate-100 overflow-hidden rounded-3xl border border-slate-200 bg-white">
          {visible.map((item) => (
            <ReadinessRow key={item.check} item={item} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={ShieldCheck}
          title="No readiness report"
          description="Run the readiness build to generate alpha_readiness_report.parquet."
        />
      )}
    </Card>
  );
}

function ReadinessRow({ item }: { item: ReadinessCheck }) {
  return (
    <div className="flex items-start gap-4 p-4 transition hover:bg-slate-50">
      <div
        className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          item.passed ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
        }`}
      >
        {item.passed ? (
          <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
        ) : (
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm font-semibold text-slate-900">{humanize(item.check)}</p>
          <span className="font-mono text-xs text-slate-500">{String(item.value || 'ok')}</span>
        </div>
        <p className="mt-1 text-sm leading-5 text-slate-500">{item.detail}</p>
      </div>
    </div>
  );
}

function SectionHeader({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon: typeof Activity;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-slate-100 text-slate-700">
          <Icon className="h-5 w-5" aria-hidden="true" />
        </div>
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-slate-950">{title}</h2>
          <p className="mt-1 text-sm leading-6 text-slate-500">{description}</p>
        </div>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <article className="rounded-[28px] border border-white/80 bg-white p-5 shadow-[0_18px_70px_rgba(15,23,42,0.07)] sm:p-6">
      {children}
    </article>
  );
}

function StatusPill({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone: 'success' | 'warning' | 'neutral';
}) {
  const classes = {
    success: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    warning: 'border-amber-200 bg-amber-50 text-amber-800',
    neutral: 'border-slate-200 bg-slate-50 text-slate-700',
  }[tone];

  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${classes}`}>
      {children}
    </span>
  );
}

function EmptyState({
  icon: Icon,
  title,
  description,
  compact = false,
}: {
  icon: typeof Activity;
  title: string;
  description: string;
  compact?: boolean;
}) {
  return (
    <div className={`mt-5 rounded-3xl border border-dashed border-slate-300 bg-slate-50 text-center ${compact ? 'p-5' : 'p-8'}`}>
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-slate-500 shadow-sm">
        <Icon className="h-5 w-5" aria-hidden="true" />
      </div>
      <h3 className="mt-4 text-sm font-semibold text-slate-900">{title}</h3>
      <p className="mx-auto mt-1 max-w-sm text-sm leading-6 text-slate-500">{description}</p>
    </div>
  );
}

function LoadingScreen() {
  return (
    <main className="min-h-screen bg-[#f6f8fb] px-4 py-5 sm:px-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-7xl">
        <div className="rounded-[28px] border border-white/80 bg-white p-5 shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
          <div className="h-4 w-36 animate-pulse rounded-full bg-slate-100" />
          <div className="mt-4 h-8 w-80 max-w-full animate-pulse rounded-full bg-slate-100" />
          <div className="mt-3 h-4 w-[32rem] max-w-full animate-pulse rounded-full bg-slate-100" />
        </div>
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="h-40 animate-pulse rounded-3xl bg-white shadow-sm" />
          ))}
        </div>
        <div className="mt-6 h-[430px] animate-pulse rounded-[28px] bg-white shadow-sm" />
      </div>
    </main>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f6f8fb] p-6">
      <div className="w-full max-w-lg rounded-[28px] border border-red-100 bg-white p-6 text-center shadow-[0_24px_80px_rgba(15,23,42,0.08)]">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-red-50 text-red-700">
          <AlertCircle className="h-6 w-6" aria-hidden="true" />
        </div>
        <h1 className="mt-5 text-xl font-semibold tracking-tight text-slate-950">Dashboard data did not load</h1>
        <p className="mt-2 text-sm leading-6 text-slate-600">{message}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-6 inline-flex items-center justify-center gap-2 rounded-full bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white transition hover:-translate-y-0.5 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        >
          <RefreshCw className="h-4 w-4" aria-hidden="true" />
          Try again
        </button>
      </div>
    </main>
  );
}

function LegendItem() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs font-medium text-slate-500">
      <span className="inline-flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-full bg-blue-600" />
        Strategy
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="h-2.5 w-2.5 rounded-full bg-slate-500" />
        SPY
      </span>
      <ChevronRight className="hidden h-4 w-4 text-slate-300 sm:block" aria-hidden="true" />
    </div>
  );
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%`;
}

function formatTime(value: Date): string {
  return value.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function formatChartDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString([], { month: 'short', year: '2-digit' });
}

function formatLongDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
}

function humanize(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
