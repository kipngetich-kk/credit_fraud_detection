import { useState, useEffect } from 'react'

interface DashboardData {
  total_findings: number
  critical_findings: number
  total_at_risk: number
  total_recovered: number
  recovery_rate: number
  findings_by_type: Record<string, number>
  top_vendors: Array<{
    name: string
    risk_score: number
    findings: number
    at_risk: number
  }>
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'findings' | 'vendors' | 'reports'>('dashboard')
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/dashboard')
      const data = await response.json()
      setDashboardData(data)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)
  }

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(1)}%`
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-navy-900 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">AP</span>
                </div>
                <span className="text-xl font-semibold text-navy-900">Guardian</span>
              </div>
              <span className="text-slate-400">|</span>
              <span className="text-sm text-slate-500">Accounts Payable Risk & Recovery</span>
            </div>
            <div className="flex items-center gap-4">
              <nav className="flex gap-1">
                {(['dashboard', 'findings', 'vendors', 'reports'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeTab === tab
                        ? 'bg-navy-900 text-white'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </nav>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-navy-900"></div>
          </div>
        ) : (
          <>
            {activeTab === 'dashboard' && dashboardData && (
              <Dashboard
                data={dashboardData}
                formatCurrency={formatCurrency}
                formatPercent={formatPercent}
              />
            )}
            {activeTab === 'findings' && <FindingsSection />}
            {activeTab === 'vendors' && dashboardData && (
              <VendorsSection vendors={dashboardData.top_vendors} formatCurrency={formatCurrency} />
            )}
            {activeTab === 'reports' && <ReportsSection />}
          </>
        )}
      </main>
    </div>
  )
}

function Dashboard({
  data,
  formatCurrency,
  formatPercent,
}: {
  data: DashboardData
  formatCurrency: (amount: number) => string
  formatPercent: (value: number) => string
}) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Findings"
          value={data.total_findings.toString()}
          subtitle={`${data.critical_findings} critical`}
          trend="up"
          color="navy"
        />
        <MetricCard
          title="At-Risk Amount"
          value={formatCurrency(data.total_at_risk)}
          subtitle="Total exposure identified"
          trend="neutral"
          color="amber"
        />
        <MetricCard
          title="Recovered"
          value={formatCurrency(data.total_recovered)}
          subtitle={`${formatPercent(data.recovery_rate)} recovery rate`}
          trend="up"
          color="green"
        />
        <MetricCard
          title="Outstanding"
          value={formatCurrency(data.total_at_risk - data.total_recovered)}
          subtitle="Pending recovery"
          trend="neutral"
          color="slate"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <FindingsByTypeChart data={data.findings_by_type} />
        </div>
        <div>
          <TopVendorsCard vendors={data.top_vendors} formatCurrency={formatCurrency} />
        </div>
      </div>

      <QuickActions />
    </div>
  )
}

function MetricCard({
  title,
  value,
  subtitle,
  trend,
  color,
}: {
  title: string
  value: string
  subtitle: string
  trend: 'up' | 'down' | 'neutral'
  color: 'navy' | 'amber' | 'green' | 'slate'
}) {
  const iconBgClasses = {
    navy: 'bg-navy-100 text-navy-900',
    amber: 'bg-amber-100 text-amber-700',
    green: 'bg-emerald-100 text-emerald-700',
    slate: 'bg-slate-100 text-slate-700',
  }

  return (
    <div className="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500 font-medium">{title}</p>
          <p className="text-3xl font-bold text-navy-900 mt-2">{value}</p>
          <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
        </div>
        <div className={`w-10 h-10 rounded-lg ${iconBgClasses[color]} flex items-center justify-center`}>
          {trend === 'up' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
          )}
          {trend === 'neutral' && (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          )}
        </div>
      </div>
    </div>
  )
}

function FindingsByTypeChart({ data }: { data: Record<string, number> }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0)
  const sortedData = Object.entries(data).sort((a, b) => b[1] - a[1])

  return (
    <div className="metric-card">
      <h3 className="text-lg font-semibold text-navy-900 mb-4">Findings by Type</h3>
      <div className="space-y-3">
        {sortedData.map(([type, count]) => {
          const percentage = (count / total) * 100
          return (
            <div key={type} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-slate-600 capitalize">{type.replace(/_/g, ' ')}</span>
                <span className="font-medium text-navy-900">{count}</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-blue-600 to-navy-900 rounded-full h-2 transition-all duration-500"
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function TopVendorsCard({
  vendors,
  formatCurrency,
}: {
  vendors: DashboardData['top_vendors']
  formatCurrency: (amount: number) => string
}) {
  return (
    <div className="metric-card">
      <h3 className="text-lg font-semibold text-navy-900 mb-4">Top Risk Vendors</h3>
      <div className="space-y-3">
        {vendors.map((vendor) => (
          <div key={vendor.name} className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm font-medium ${
                vendor.risk_score >= 70
                  ? 'bg-red-100 text-red-700'
                  : vendor.risk_score >= 40
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-slate-100 text-slate-700'
              }`}
            >
              {vendor.risk_score}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-navy-900 truncate">{vendor.name}</p>
              <p className="text-xs text-slate-500">
                {vendor.findings} findings · {formatCurrency(vendor.at_risk)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function QuickActions() {
  return (
    <div className="metric-card">
      <h3 className="text-lg font-semibold text-navy-900 mb-4">Quick Actions</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
          <div className="w-10 h-10 bg-navy-900 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
          </div>
          <div className="text-left">
            <p className="font-medium text-navy-900">New Audit</p>
            <p className="text-sm text-slate-500">Start a new scan</p>
          </div>
        </button>
        <button className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
          </div>
          <div className="text-left">
            <p className="font-medium text-navy-900">Export Data</p>
            <p className="text-sm text-slate-500">Download to Excel</p>
          </div>
        </button>
        <button className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
          <div className="w-10 h-10 bg-emerald-600 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div className="text-left">
            <p className="font-medium text-navy-900">Validate Findings</p>
            <p className="text-sm text-slate-500">Review queue: 12</p>
          </div>
        </button>
      </div>
    </div>
  )
}

function FindingsSection() {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-navy-900">Findings</h2>
          <p className="text-slate-500">Review and manage identified risks</p>
        </div>
        <div className="flex gap-2">
          <select className="border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white">
            <option>All Severities</option>
            <option>Critical</option>
            <option>High</option>
            <option>Medium</option>
            <option>Low</option>
          </select>
          <select className="border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white">
            <option>All Types</option>
            <option>Duplicate Payment</option>
            <option>Threshold Anomaly</option>
            <option>Vendor Risk</option>
            <option>Compliance Gap</option>
          </select>
        </div>
      </div>

      <div className="metric-card overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                ID
              </th>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Vendor
              </th>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Type
              </th>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Severity
              </th>
              <th className="text-right text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Amount
              </th>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
              <tr key={i} className="hover:bg-slate-50 transition-colors">
                <td className="px-4 py-3 text-sm font-medium text-navy-900">F2026-{String(i).padStart(4, '0')}</td>
                <td className="px-4 py-3 text-sm text-slate-600">Vendor {i}</td>
                <td className="px-4 py-3 text-sm text-slate-600">
                  {['Duplicate', 'Threshold', 'Vendor Risk', 'Compliance'][i % 4]}
                </td>
                <td className="px-4 py-3">
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      i % 4 === 0
                        ? 'severity-critical'
                        : i % 4 === 1
                        ? 'severity-high'
                        : i % 4 === 2
                        ? 'severity-medium'
                        : 'severity-low'
                    }`}
                  >
                    {['Critical', 'High', 'Medium', 'Low'][i % 4]}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-navy-900 font-medium text-right">
                  ${(500 + i * 500).toLocaleString()}
                </td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-700">
                    Open
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function VendorsSection({
  vendors,
  formatCurrency,
}: {
  vendors: DashboardData['top_vendors']
  formatCurrency: (amount: number) => string
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-navy-900">Vendor Risk Scorecard</h2>
        <p className="text-slate-500">Risk scoring based on finding patterns and exposure</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="metric-card">
          <p className="text-sm text-slate-500">High Risk</p>
          <p className="text-3xl font-bold text-red-600 mt-1">
            {vendors.filter((v) => v.risk_score >= 70).length}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-slate-500">Medium Risk</p>
          <p className="text-3xl font-bold text-amber-600 mt-1">
            {vendors.filter((v) => v.risk_score >= 40 && v.risk_score < 70).length}
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm text-slate-500">Low Risk</p>
          <p className="text-3xl font-bold text-slate-600 mt-1">
            {vendors.filter((v) => v.risk_score < 40).length}
          </p>
        </div>
      </div>

      <div className="metric-card overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="text-left text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Vendor
              </th>
              <th className="text-center text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Risk Score
              </th>
              <th className="text-center text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                Findings
              </th>
              <th className="text-right text-xs font-semibold text-slate-500 uppercase tracking-wider px-4 py-3">
                At-Risk
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {vendors.map((vendor) => (
              <tr key={vendor.name} className="hover:bg-slate-50 transition-colors">
                <td className="px-4 py-4">
                  <div>
                    <p className="font-medium text-navy-900">{vendor.name}</p>
                  </div>
                </td>
                <td className="px-4 py-4 text-center">
                  <div
                    className={`inline-flex items-center justify-center w-12 h-8 rounded-lg font-bold ${
                      vendor.risk_score >= 70
                        ? 'bg-red-100 text-red-700'
                        : vendor.risk_score >= 40
                        ? 'bg-amber-100 text-amber-700'
                        : 'bg-slate-100 text-slate-700'
                    }`}
                  >
                    {vendor.risk_score}
                  </div>
                </td>
                <td className="px-4 py-4 text-center text-slate-600">{vendor.findings}</td>
                <td className="px-4 py-4 text-right font-medium text-navy-900">
                  {formatCurrency(vendor.at_risk)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ReportsSection() {
  const [isDownloading, setIsDownloading] = useState(false)

  const handleDownloadPDF = async () => {
    setIsDownloading(true)
    try {
      const response = await fetch('/api/reports/demo/download', {
        method: 'GET',
      })

      if (!response.ok) {
        throw new Error('Failed to generate report')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'AP_Guardian_Audit_Report.pdf'
      document.body.appendChild(a)
      a.click()

      window.URL.revokeObjectURL(url)
      a.remove()
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download report. Please try again.')
    } finally {
      setIsDownloading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-navy-900">Export Center</h2>
        <p className="text-slate-500">Generate and download audit reports</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <div className="flex flex-col items-center text-center py-6">
            <div className="w-16 h-16 bg-gradient-to-br from-navy-700 to-navy-900 rounded-xl flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-navy-900 mb-2">Executive PDF Report</h3>
            <p className="text-sm text-slate-500 mb-6">
              Professional audit report designed for CFO and management review
            </p>
            <button
              onClick={handleDownloadPDF}
              disabled={isDownloading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {isDownloading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Generating...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Download PDF Report
                </>
              )}
            </button>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex flex-col items-center text-center py-6">
            <div className="w-16 h-16 bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-xl flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-navy-900 mb-2">Excel Export</h3>
            <p className="text-sm text-slate-500 mb-6">
              Raw data export for further analysis and processing
            </p>
            <button className="btn-secondary w-full flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Download Excel
            </button>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex flex-col items-center text-center py-6">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-navy-900 mb-2">View Online</h3>
            <p className="text-sm text-slate-500 mb-6">
              Interactive dashboard view with drill-down capabilities
            </p>
            <button className="btn-secondary w-full flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              View Dashboard
            </button>
          </div>
        </div>
      </div>

      <div className="metric-card">
        <h3 className="text-lg font-semibold text-navy-900 mb-4">Recent Reports</h3>
        <div className="space-y-3">
          {[
            { name: 'Q1 2026 Audit Report', date: 'March 15, 2026', type: 'PDF' },
            { name: 'Vendor Risk Analysis - February', date: 'February 28, 2026', type: 'Excel' },
            { name: 'Q4 2025 Audit Report', date: 'December 20, 2025', type: 'PDF' },
          ].map((report, i) => (
            <div
              key={i}
              className="flex items-center justify-between p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    report.type === 'PDF' ? 'bg-red-100' : 'bg-emerald-100'
                  }`}
                >
                  <svg className={`w-5 h-5 ${report.type === 'PDF' ? 'text-red-600' : 'text-emerald-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div>
                  <p className="font-medium text-navy-900">{report.name}</p>
                  <p className="text-sm text-slate-500">{report.date}</p>
                </div>
              </div>
              <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                Download
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
