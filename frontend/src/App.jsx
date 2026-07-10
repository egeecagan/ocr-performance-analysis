import { useState } from 'react'
import Dashboard from './pages/Dashboard'
import Scanner   from './pages/Scanner'
import './index.css'

export default function App() {
  const [tab, setTab] = useState('scanner')

  return (
    <div className="app-shell">
      {/* ── Topbar ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <path d="M9 9h6M9 12h6M9 15h4"/>
          </svg>
          OCR Performance Analysis
        </div>

        <nav className="topbar-nav">
          <button
            className={`nav-btn ${tab === 'scanner' ? 'active' : ''}`}
            onClick={() => setTab('scanner')}
          >
            🔍 Belge Tarama
          </button>
          <button
            className={`nav-btn ${tab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setTab('dashboard')}
          >
            📊 Karşılaştırma
          </button>
        </nav>
      </header>

      {/* ── İçerik ── */}
      <main className="page">
        {tab === 'scanner'   && <Scanner />}
        {tab === 'dashboard' && <Dashboard />}
      </main>
    </div>
  )
}
