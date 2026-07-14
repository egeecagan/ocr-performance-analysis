import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import Scanner from './pages/Scanner'
import ReportsHistory from './pages/ReportsHistory'
import './index.css'

export default function App() {
  const [tab, setTab] = useState('scanner')

  return (
    <div className="app-shell">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          OCR Performance Analysis
        </div>

        <nav className="topbar-nav">
          <button
            className={`nav-btn ${tab === 'scanner' ? 'active' : ''}`}
            onClick={() => setTab('scanner')}
          >
            Belge Tarama
          </button>
          <button
            className={`nav-btn ${tab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setTab('dashboard')}
          >
            Karşılaştırma
          </button>
          <button
            className={`nav-btn ${tab === 'history' ? 'active' : ''}`}
            onClick={() => setTab('history')}
          >
            Geçmiş Raporlar
          </button>
        </nav>
      </header>

      {/* ── İçerik ── */}
      <main className="page">
        {tab === 'scanner' && <Scanner />}
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'history' && <ReportsHistory />}
      </main>
    </div>
  )
}
