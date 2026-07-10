/**
 * Dashboard.jsx — Model Karşılaştırma Raporu
 *
 * /report endpoint'inden comparison_report.json'ı okur.
 * - Model bazlı KPI kartları
 * - Çubuk grafik: CER / WER / Confidence karşılaştırması
 * - Radar grafik: çok boyutlu skor görünümü
 */

import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

const API = 'http://localhost:8000'

// ── Renkler ────────────────────────────────────────────────────────────────
const PALETTE = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4']

// ── Yardimci: rapor verisinden metrik listesi cikar ───────────────────────
/**
 * comparison_report.json yapısı:
 * {
 *   "dekont": {
 *     "tesseract/model_v1": {
 *       "avg_total_time_seconds": 0.57,
 *       "avg_confidence": 66.58,
 *       "common_fields": { "avg_cer": 0.23, "avg_wer": 0.46, ... },
 *       "specific_keyword_success_rates": { "DEKONT": 80, ... }
 *     }
 *   },
 *   "surucubelgesi": {
 *     "tesseract/model_v1": {
 *       "avg_cer": 0.32,          ← üst seviyede de olabilir
 *       "avg_wer": 0.61,
 *       "avg_total_time_seconds": 0.51,
 *       "avg_confidence": 48.03,
 *       "avg_field_match_ratio": 0.40,
 *       "common_fields": { "avg_cer": 0.18, ... },
 *       "specific_keyword_success_rates": { ... }
 *     }
 *   }
 * }
 */
function extractMetrics(report) {
  if (!report || typeof report !== 'object') return []

  const rows = []

  // Üst seviye: belge tipi (dekont, surucubelgesi ...)
  Object.entries(report).forEach(([docType, models]) => {
    if (typeof models !== 'object') return

    // İkinci seviye: "engine/model_name" → metrikler
    Object.entries(models).forEach(([engineModel, stats]) => {
      if (typeof stats !== 'object') return

      // engine/model_name ayrıştır ("tesseract/model_v1" → ["tesseract","model_v1"])
      const slashIdx = engineModel.indexOf('/')
      const engine   = slashIdx > -1 ? engineModel.slice(0, slashIdx) : engineModel
      const model    = slashIdx > -1 ? engineModel.slice(slashIdx + 1) : ''

      // CER/WER: önce üst seviyede ara, yoksa common_fields'a bak
      const cf = stats.common_fields ?? {}
      const cer  = stats.avg_cer  ?? cf.avg_cer  ?? null
      const wer  = stats.avg_wer  ?? cf.avg_wer  ?? null
      const conf = stats.avg_confidence ?? null
      const speed = stats.avg_total_time_seconds ?? null

      // Hit rate: specific_keyword_success_rates'in ortalaması
      const kwRates = stats.specific_keyword_success_rates ?? {}
      const kwVals  = Object.values(kwRates).filter(v => typeof v === 'number')
      const hitRate = kwVals.length
        ? kwVals.reduce((a, b) => a + b, 0) / kwVals.length
        : null

      // Field match ratio (sürücü belgesi için)
      const fieldMatch = stats.avg_field_match_ratio ?? cf.avg_common_field_match_ratio ?? null

      rows.push({
        label:      `${docType} · ${engine}/${model}`,
        docType,
        engine,
        model,
        cer:        cer      != null ? +(cer * 100).toFixed(2)  : null,
        wer:        wer      != null ? +(wer * 100).toFixed(2)  : null,
        conf:       conf     != null ? +conf.toFixed(2)          : null,
        hitRate:    hitRate  != null ? +hitRate.toFixed(2)       : null,
        fieldMatch: fieldMatch != null ? +(fieldMatch * 100).toFixed(2) : null,
        speed:      speed    != null ? +speed.toFixed(3)         : null,
        rawStats: stats,
      })
    })
  })

  return rows
}

// ── KPI Kart ─────────────────────────────────────────────────────────────
function KpiCard({ label, value, unit = '', color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={color ? { color } : {}}>
        {value ?? '—'}{value != null && <span className="kpi-unit">{unit}</span>}
      </div>
    </div>
  )
}

// ── Özel tooltip ──────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-card)',
      border: '1px solid var(--border-focus)',
      borderRadius: '10px',
      padding: '0.75rem 1rem',
      fontSize: '0.8rem',
      color: 'var(--text-primary)',
      boxShadow: 'var(--shadow-lg)',
    }}>
      <p style={{ marginBottom: '0.4rem', fontWeight: 600 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>
          {p.name}: <strong>{p.value}</strong>
        </p>
      ))}
    </div>
  )
}

// ── Ana bileşen ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [report,  setReport]  = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [selected, setSelected] = useState(null) // secili model indexi

  useEffect(() => {
    axios.get(`${API}/report`)
      .then(r => {
        setReport(r.data)
        setLoading(false)
      })
      .catch(err => {
        setError(
          err.response?.status === 404
            ? 'Rapor henüz oluşturulmamış. Önce main.py\'i çalıştırın.'
            : 'API\'ye ulaşılamadı. Uvicorn sunucusunun çalıştığını kontrol edin.'
        )
        setLoading(false)
      })
  }, [])

  if (loading) return (
    <div className="spinner-wrap"><div className="spinner" /><span>Rapor yükleniyor...</span></div>
  )

  if (error) return (
    <div className="empty-state">
      <h3>⚠ Rapor Bulunamadı</h3>
      <p style={{ marginTop: '0.5rem' }}>{error}</p>
    </div>
  )

  const rows = extractMetrics(report)

  if (rows.length === 0) return (
    <div className="empty-state">
      <h3>Raporda veri yok</h3>
      <p>comparison_report.json boş veya beklenen formatta değil.</p>
    </div>
  )

  const sel = selected != null ? rows[selected] : null

  // Grafik verisi
  const barData = rows.map((r, i) => ({
    name:    `${r.docType}·${r.model}`,
    engine:  r.engine,
    'CER %': r.cer,
    'WER %': r.wer,
    'Güven': r.conf,
    color:   PALETTE[i % PALETTE.length],
  }))

  const radarData = [
    { metric: 'CER (düşük=iyi)', ...Object.fromEntries(rows.map(r => [r.label, r.cer])) },
    { metric: 'WER (düşük=iyi)', ...Object.fromEntries(rows.map(r => [r.label, r.wer])) },
    { metric: 'Güven %',         ...Object.fromEntries(rows.map(r => [r.label, r.conf])) },
    { metric: 'Hit Rate %',      ...Object.fromEntries(rows.map(r => [r.label, r.hitRate])) },
    { metric: 'Hız (s, düşük=iyi)', ...Object.fromEntries(rows.map(r => [r.label, r.speed])) },
  ].filter(d => Object.values(d).some(v => v != null && typeof v === 'number'))

  return (
    <div>
      <h2 className="section-heading">📊 Model Karşılaştırma Raporu</h2>

      {/* ── Model chip seçici ── */}
      <div className="model-select-bar">
        {rows.map((r, i) => (
          <button
            key={i}
            className={`model-chip ${selected === i ? 'active' : ''}`}
            onClick={() => setSelected(selected === i ? null : i)}
          >
            <span style={{
              display: 'inline-block',
              width: 8, height: 8,
              borderRadius: '50%',
              background: PALETTE[i % PALETTE.length],
              marginRight: 6,
            }} />
            {r.docType} · {r.engine}/{r.model}
          </button>
        ))}
      </div>

      {/* ── Seçili modelin KPI kartları ── */}
      {sel ? (
        <div className="kpi-grid" style={{ marginBottom: '1.5rem' }}>
          <KpiCard label="CER" value={sel.cer} unit="%"
            color={sel.cer == null ? undefined : sel.cer < 10 ? 'var(--green)' : sel.cer < 30 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="WER" value={sel.wer} unit="%"
            color={sel.wer == null ? undefined : sel.wer < 15 ? 'var(--green)' : sel.wer < 40 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="Ortalama Güven" value={sel.conf} unit="%"
            color={sel.conf == null ? undefined : sel.conf >= 80 ? 'var(--green)' : sel.conf >= 50 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="Hit Rate" value={sel.hitRate} unit="%" />
          <KpiCard label="Alan Eşleşme" value={sel.fieldMatch} unit="%" color="var(--accent-light)" />
          <KpiCard label="Ort. Hız" value={sel.speed} unit="s" />
        </div>
      ) : (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '1.5rem' }}>
          Detaylı metrikleri görmek için yukarıdan bir model seçin.
        </p>
      )}

      {/* ── Grafikler ── */}
      <div className="dashboard-grid">

        {/* Çubuk Grafik */}
        <div className="card">
          <div className="card-title">CER / WER / Güven Karşılaştırması</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,172,255,0.08)" />
              <XAxis dataKey="name" tick={{ fill: '#8fafd4', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8fafd4', fontSize: 11 }} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.78rem', color: '#8fafd4' }} />
              <Bar dataKey="CER %" fill="#ef4444" radius={[4,4,0,0]} />
              <Bar dataKey="WER %" fill="#f59e0b" radius={[4,4,0,0]} />
              <Bar dataKey="Güven" fill="#3b82f6" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Grafik */}
        <div className="card">
          <div className="card-title">Çok Boyutlu Model Karşılaştırması</div>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(99,172,255,0.15)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8fafd4', fontSize: 10 }} />
                {rows.map((r, i) => (
                  <Radar
                    key={r.label}
                    name={`${r.engine}/${r.model}`}
                    dataKey={r.label}
                    stroke={PALETTE[i % PALETTE.length]}
                    fill={PALETTE[i % PALETTE.length]}
                    fillOpacity={0.12}
                  />
                ))}
                <Legend wrapperStyle={{ fontSize: '0.78rem', color: '#8fafd4' }} />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state" style={{ padding: '2rem' }}>
              <p>Radar için yeterli metrik verisi yok.</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Tam tablo ── */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-title">Tüm Model Metrikleri</div>
        <table className="field-table">
          <thead>
            <tr>
              <th>Belge Tipi</th>
              <th>Engine / Model</th>
              <th>CER %</th>
              <th>WER %</th>
              <th>Güven %</th>
              <th>Hit Rate %</th>
              <th>Alan Eşl. %</th>
              <th>Hız (s)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={selected === i ? { background: 'var(--bg-card-hover)' } : {}}>
                <td>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8,
                    borderRadius: '50%',
                    background: PALETTE[i % PALETTE.length],
                    marginRight: 6,
                  }} />
                  {r.docType}
                </td>
                <td style={{ color: 'var(--text-secondary)' }}>{r.engine} / {r.model}</td>
                <td style={{ color: r.cer == null ? 'var(--text-muted)' : r.cer < 10 ? 'var(--green)' : r.cer < 30 ? 'var(--yellow)' : 'var(--red)' }}>
                  {r.cer ?? '—'}
                </td>
                <td style={{ color: r.wer == null ? 'var(--text-muted)' : r.wer < 15 ? 'var(--green)' : r.wer < 40 ? 'var(--yellow)' : 'var(--red)' }}>
                  {r.wer ?? '—'}
                </td>
                <td style={{ color: r.conf == null ? 'var(--text-muted)' : r.conf >= 80 ? 'var(--green)' : r.conf >= 50 ? 'var(--yellow)' : 'var(--red)' }}>
                  {r.conf ?? '—'}
                </td>
                <td>{r.hitRate != null ? `${r.hitRate}%` : '—'}</td>
                <td>{r.fieldMatch != null ? `${r.fieldMatch}%` : '—'}</td>
                <td>{r.speed ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
