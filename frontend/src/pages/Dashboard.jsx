/**
 * Dashboard.jsx — Model Karşılaştırma Raporu
 *
 * /report endpoint'inden comparison_report.json'ı okur.
 * - Model bazlı KPI kartları
 * - Çubuk grafik: CER / WER / Confidence karşılaştırması
 * - Radar grafik: çok boyutlu skor görünümü
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import ImageViewer, { ConfidenceLegend } from '../components/ImageViewer'
import ResultsPanel from '../components/ResultsPanel'
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
// ── Ana bileşen ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [report,  setReport]  = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)
  const [selected, setSelected] = useState(null) // secili model indexi
  const [docType, setDocType] = useState('surucubelgesi') // surucubelgesi veya dekont

  // Pipeline durumları
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineProgress, setPipelineProgress] = useState('')
  const [pipelineError, setPipelineError] = useState(null)

  const pollIntervalRef = useRef(null)

  const loadReport = useCallback(() => {
    setLoading(true)
    axios.get(`${API}/report`)
      .then(r => {
        setReport(r.data)
        setError(null)
        setLoading(false)
      })
      .catch(err => {
        setError(
          err.response?.status === 404
            ? 'Rapor henüz oluşturulmamış. Önce "Tüm Modelleri Çalıştır ve Yeniden Raporla" butonuna basın.'
            : 'API\'ye ulaşılamadı. Uvicorn sunucusunun çalıştığını kontrol edin.'
        )
        setLoading(false)
      })
  }, [])

  const startPolling = useCallback(() => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    pollIntervalRef.current = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}/pipeline-status`)
        if (data.status === 'running') {
          setPipelineRunning(true)
          setPipelineProgress(data.progress)
          setPipelineError(null)
        } else if (data.status === 'success') {
          setPipelineRunning(false)
          setPipelineProgress('')
          setPipelineError(null)
          clearInterval(pollIntervalRef.current)
          loadReport()
        } else if (data.status === 'error') {
          setPipelineRunning(false)
          setPipelineProgress('')
          setPipelineError(data.error)
          clearInterval(pollIntervalRef.current)
        } else {
          // idle
          setPipelineRunning(false)
          clearInterval(pollIntervalRef.current)
        }
      } catch (err) {
        console.error("Pipeline status fetch failed:", err)
      }
    }, 1500)
  }, [loadReport])

  useEffect(() => {
    // İlk olarak arka plandaki pipeline durumunu kontrol et
    axios.get(`${API}/pipeline-status`)
      .then(r => {
        if (r.data.status === 'running') {
          setPipelineRunning(true)
          setPipelineProgress(r.data.progress)
          startPolling()
        } else {
          loadReport()
        }
      })
      .catch(() => {
        loadReport()
      })

    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [startPolling, loadReport])

  const handleRunPipeline = async () => {
    setPipelineRunning(true)
    setPipelineProgress('Başlatılıyor...')
    setPipelineError(null)
    try {
      await axios.post(`${API}/run-pipeline`)
      startPolling()
    } catch (err) {
      setPipelineError(err.response?.data?.detail ?? 'Pipeline başlatılırken hata oluştu.')
      setPipelineRunning(false)
    }
  }

  const rows = report ? extractMetrics(report) : []
  const filteredRows = rows.filter(r => r.docType === docType)
  const sel = selected != null ? filteredRows[selected] : null


  // Grafik verisi
  const barData = filteredRows.map((r, i) => ({
    name:    r.model === 'model_v1' ? r.engine : `${r.engine}/${r.model}`,
    engine:  r.engine,
    'CER %': r.cer,
    'WER %': r.wer,
    'Güven': r.conf,
    color:   PALETTE[i % PALETTE.length],
  }))

  const radarData = [
    { metric: 'CER (düşük=iyi)', ...Object.fromEntries(filteredRows.map(r => [`${r.engine}/${r.model}`, r.cer])) },
    { metric: 'WER (düşük=iyi)', ...Object.fromEntries(filteredRows.map(r => [`${r.engine}/${r.model}`, r.wer])) },
    { metric: 'Güven %',         ...Object.fromEntries(filteredRows.map(r => [`${r.engine}/${r.model}`, r.conf])) },
    { metric: 'Hit Rate %',      ...Object.fromEntries(filteredRows.map(r => [`${r.engine}/${r.model}`, r.hitRate])) },
    { metric: 'Hız (s, düşük=iyi)', ...Object.fromEntries(filteredRows.map(r => [`${r.engine}/${r.model}`, r.speed])) },
  ].filter(d => Object.values(d).some(v => v != null && typeof v === 'number'))

  return (
    <div>
      {/* Header controls with title and Run Pipeline button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h2 className="section-heading" style={{ margin: 0 }}>Karşılaştırma & Performans Analizi</h2>
        <button
          className="btn btn-primary"
          onClick={handleRunPipeline}
          disabled={pipelineRunning}
          style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}
        >
          {pipelineRunning ? '⏳ Toplu Rapor Üretiliyor...' : '⚡ Tüm Modelleri Çalıştır ve Yeniden Raporla'}
        </button>
      </div>

      {pipelineError && (
        <div className="error-box" style={{ marginBottom: '1.5rem' }}>
          ⚠ {pipelineError}
        </div>
      )}

      {/* ── SEKMELER / DURUMLAR ── */}
      {pipelineRunning ? (
        <div className="spinner-wrap" style={{ margin: '4rem 0' }}>
          <div className="spinner" />
          <span style={{ fontWeight: 500 }}>{pipelineProgress}</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Bu işlem tüm OCR modellerini veri tabanındaki tüm dosyalarla çalıştırdığı için 1-3 dakika sürebilir.
          </span>
        </div>
      ) : loading ? (
        <div className="spinner-wrap"><div className="spinner" /><span>Rapor yükleniyor...</span></div>
      ) : error ? (
        <div className="empty-state">
          <h3>⚠ Rapor Bulunamadı</h3>
          <p style={{ marginTop: '0.5rem' }}>{error}</p>
        </div>
      ) : rows.length === 0 ? (
        <div className="empty-state">
          <h3>Raporda veri yok</h3>
          <p>comparison_report.json boş veya beklenen formatta değil.</p>
        </div>
      ) : (
        <div>
          {/* ── Alt Sekme Seçici ── */}
          <div className="sub-tabs" style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <button
              className={`nav-btn ${docType === 'surucubelgesi' ? 'active' : ''}`}
              onClick={() => { setDocType('surucubelgesi'); setSelected(null); }}
              style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
            >
              🪪 Sürücü Belgesi Karşılaştırması
            </button>
            <button
              className={`nav-btn ${docType === 'dekont' ? 'active' : ''}`}
              onClick={() => { setDocType('dekont'); setSelected(null); }}
              style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
            >
              🧾 Dekont Karşılaştırması
            </button>
          </div>

          {/* Model chip seçici */}
          <div className="model-select-bar">
            {filteredRows.map((r, i) => (
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
                {r.engine}/{r.model}
              </button>
            ))}
          </div>

          {/* Seçili modelin KPI kartları */}
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

          {/* Grafikler */}
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
                    {filteredRows.map((r, i) => (
                      <Radar
                        key={r.label}
                        name={`${r.engine}/${r.model}`}
                        dataKey={`${r.engine}/${r.model}`}
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

          {/* Tam tablo */}
          <div className="card" style={{ marginTop: '1.5rem' }}>
            <div className="card-title">{docType === 'surucubelgesi' ? '🪪 Sürücü Belgesi Model Metrikleri' : '🧾 Dekont Model Metrikleri'}</div>
            <table className="field-table">
              <thead>
                <tr>
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
                {filteredRows.map((r, i) => (
                  <tr key={i} style={selected === i ? { background: 'var(--bg-card-hover)' } : {}}>
                    <td>
                      <span style={{
                        display: 'inline-block', width: 8, height: 8,
                        borderRadius: '50%',
                        background: PALETTE[i % PALETTE.length],
                        marginRight: 6,
                      }} />
                      {r.engine} / {r.model}
                    </td>
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
      )}
    </div>
  )
}
