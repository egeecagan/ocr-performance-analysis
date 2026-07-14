/**
 * Dashboard.jsx — Özel Karşılaştırma
 *
 * Kullanıcı istediği görselleri ve model/config kombinasyonlarını seçerek
 * karşılaştırma çalıştırır. Yeni config eklendiğinde otomatik listelenir.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'

const API = 'http://localhost:8000'

const PALETTE = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#ef4444', '#06b6d4']

// ── Yardimci: rapor verisinden metrik listesi cikar ───────────────────────
function extractMetrics(report) {
  if (!report || typeof report !== 'object') return []
  const rows = []
  Object.entries(report).forEach(([docType, models]) => {
    if (typeof models !== 'object') return
    Object.entries(models).forEach(([engineModel, stats]) => {
      if (typeof stats !== 'object') return
      const slashIdx   = engineModel.indexOf('/')
      const engine     = slashIdx > -1 ? engineModel.slice(0, slashIdx) : engineModel
      const model      = slashIdx > -1 ? engineModel.slice(slashIdx + 1) : ''
      const cf         = stats.common_fields ?? {}
      const cer        = stats.avg_cer  ?? cf.avg_cer  ?? null
      const wer        = stats.avg_wer  ?? cf.avg_wer  ?? null
      const conf       = stats.avg_confidence ?? null
      const speed      = stats.avg_total_time_seconds ?? null
      const kwRates    = stats.specific_keyword_success_rates ?? {}
      const kwVals     = Object.values(kwRates).filter(v => typeof v === 'number')
      const hitRate    = kwVals.length ? kwVals.reduce((a, b) => a + b, 0) / kwVals.length : null
      const fieldMatch = stats.avg_field_match_ratio ?? cf.avg_common_field_match_ratio ?? null
      rows.push({
        label: `${docType} · ${engine}/${model}`,
        docType, engine, model,
        cer:        cer        != null ? +(cer * 100).toFixed(2)        : null,
        wer:        wer        != null ? +(wer * 100).toFixed(2)        : null,
        conf:       conf       != null ? +conf.toFixed(2)               : null,
        hitRate:    hitRate    != null ? +hitRate.toFixed(2)            : null,
        fieldMatch: fieldMatch != null ? +(fieldMatch * 100).toFixed(2) : null,
        speed:      speed      != null ? +speed.toFixed(3)             : null,
        rawStats: stats,
      })
    })
  })
  return rows
}

// ── KPI Kart ──────────────────────────────────────────────────────────────
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
      background: 'var(--bg-card)', border: '1px solid var(--border-focus)',
      borderRadius: '10px', padding: '0.75rem 1rem',
      fontSize: '0.8rem', color: 'var(--text-primary)', boxShadow: 'var(--shadow-lg)',
    }}>
      <p style={{ marginBottom: '0.4rem', fontWeight: 600 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: <strong>{p.value}</strong></p>
      ))}
    </div>
  )
}

// ── Grafik + Tablo ────────────────────────────────────────────────────────
function ReportCharts({ rows, docTypeLabel }) {
  const [selected, setSelected] = useState(null)
  const sel = selected != null ? rows[selected] : null

  const barData = rows.map((r, i) => ({
    name:    r.model === 'model_v1' ? r.engine : `${r.engine}/${r.model}`,
    'CER %': r.cer, 'WER %': r.wer, 'Güven': r.conf,
    color:   PALETTE[i % PALETTE.length],
  }))

  const radarData = [
    { metric: 'CER (düşük=iyi)', ...Object.fromEntries(rows.map(r => [`${r.engine}/${r.model}`, r.cer])) },
    { metric: 'WER (düşük=iyi)', ...Object.fromEntries(rows.map(r => [`${r.engine}/${r.model}`, r.wer])) },
    { metric: 'Güven %',         ...Object.fromEntries(rows.map(r => [`${r.engine}/${r.model}`, r.conf])) },
    { metric: 'Hit Rate %',      ...Object.fromEntries(rows.map(r => [`${r.engine}/${r.model}`, r.hitRate])) },
    { metric: 'Hız (s)',         ...Object.fromEntries(rows.map(r => [`${r.engine}/${r.model}`, r.speed])) },
  ].filter(d => Object.values(d).some(v => v != null && typeof v === 'number'))

  return (
    <div>
      {/* Model chip seçici */}
      <div className="model-select-bar">
        {rows.map((r, i) => (
          <button key={i} className={`model-chip ${selected === i ? 'active' : ''}`}
            onClick={() => setSelected(selected === i ? null : i)}>
            <span style={{
              display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
              background: PALETTE[i % PALETTE.length], marginRight: 6,
            }} />
            {r.engine}/{r.model}
          </button>
        ))}
      </div>

      {sel ? (
        <div className="kpi-grid" style={{ marginBottom: '1.5rem' }}>
          <KpiCard label="CER" value={sel.cer} unit="%" color={sel.cer == null ? undefined : sel.cer < 10 ? 'var(--green)' : sel.cer < 30 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="WER" value={sel.wer} unit="%" color={sel.wer == null ? undefined : sel.wer < 15 ? 'var(--green)' : sel.wer < 40 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="Ortalama Güven" value={sel.conf} unit="%" color={sel.conf == null ? undefined : sel.conf >= 80 ? 'var(--green)' : sel.conf >= 50 ? 'var(--yellow)' : 'var(--red)'} />
          <KpiCard label="Hit Rate" value={sel.hitRate} unit="%" />
          <KpiCard label="Alan Eşleşme" value={sel.fieldMatch} unit="%" color="var(--accent-light)" />
          <KpiCard label="Ort. Hız" value={sel.speed} unit="s" />
        </div>
      ) : (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '1.5rem' }}>
          Detaylı metrikleri görmek için yukarıdan bir model seçin.
        </p>
      )}

      <div className="dashboard-grid">
        <div className="card">
          <div className="card-title">CER / WER / Güven Karşılaştırması</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,172,255,0.08)" />
              <XAxis dataKey="name" tick={{ fill: '#8fafd4', fontSize: 11 }} />
              <YAxis tick={{ fill: '#8fafd4', fontSize: 11 }} unit="%" />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: '0.78rem', color: '#8fafd4' }} />
              <Bar dataKey="CER %" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="WER %" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Güven" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">Çok Boyutlu Model Karşılaştırması</div>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(99,172,255,0.15)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8fafd4', fontSize: 10 }} />
                {rows.map((r, i) => (
                  <Radar key={r.label} name={`${r.engine}/${r.model}`}
                    dataKey={`${r.engine}/${r.model}`}
                    stroke={PALETTE[i % PALETTE.length]} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.12} />
                ))}
                <Legend wrapperStyle={{ fontSize: '0.78rem', color: '#8fafd4' }} />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state" style={{ padding: '2rem' }}><p>Radar için yeterli metrik verisi yok.</p></div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-title">{docTypeLabel} — Model Metrikleri</div>
        <table className="field-table">
          <thead>
            <tr>
              <th>Engine / Model</th><th>CER %</th><th>WER %</th>
              <th>Güven %</th><th>Hit Rate %</th><th>Alan Eşl. %</th><th>Hız (s)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={selected === i ? { background: 'var(--bg-card-hover)' } : {}}>
                <td>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                    background: PALETTE[i % PALETTE.length], marginRight: 6,
                  }} />
                  {r.engine} / {r.model}
                </td>
                <td style={{ color: r.cer == null ? 'var(--text-muted)' : r.cer < 10 ? 'var(--green)' : r.cer < 30 ? 'var(--yellow)' : 'var(--red)' }}>{r.cer ?? '—'}</td>
                <td style={{ color: r.wer == null ? 'var(--text-muted)' : r.wer < 15 ? 'var(--green)' : r.wer < 40 ? 'var(--yellow)' : 'var(--red)' }}>{r.wer ?? '—'}</td>
                <td style={{ color: r.conf == null ? 'var(--text-muted)' : r.conf >= 80 ? 'var(--green)' : r.conf >= 50 ? 'var(--yellow)' : 'var(--red)' }}>{r.conf ?? '—'}</td>
                <td>{r.hitRate  != null ? `${r.hitRate}%`  : '—'}</td>
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

// ── Ana bileşen ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [engines, setEngines]            = useState({})
  const [images, setImages]              = useState([])
  const [selectedConfigs, setSelected]   = useState(new Set())
  const [dragOver, setDragOver]          = useState(false)
  const [running, setRunning]            = useState(false)
  const [progress, setProgress]          = useState('')
  const [error, setError]                = useState(null)
  const [result, setResult]              = useState(null)
  const pollRef                          = useRef(null)
  const fileInputRef                     = useRef(null)

  // Engine listesi — /engines her yeni YAML'ı otomatik içerir
  useEffect(() => {
    axios.get(`${API}/engines`).then(r => setEngines(r.data)).catch(() => {})
  }, [])

  // Tümünü seç / kaldır
  const allKeys     = Object.entries(engines).flatMap(([eng, models]) => models.map(m => `${eng}/${m}`))
  const allSelected = allKeys.length > 0 && allKeys.every(k => selectedConfigs.has(k))
  const toggleAll   = () => setSelected(allSelected ? new Set() : new Set(allKeys))
  const toggleConfig = (key) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  // Görsel ekle
  const addFiles = (files) => {
    const valid = [...files].filter(f =>
      f.type.startsWith('image/') || /\.(png|jpe?g|tiff?|bmp|webp)$/i.test(f.name)
    )
    setImages(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...valid.filter(f => !names.has(f.name))]
    })
  }
  const removeImage = (name) => setImages(prev => prev.filter(f => f.name !== name))

  // Drag-drop
  const onDragOver  = (e) => { e.preventDefault(); setDragOver(true) }
  const onDragLeave = () => setDragOver(false)
  const onDrop      = (e) => { e.preventDefault(); setDragOver(false); addFiles(e.dataTransfer.files) }

  // Polling
  const stopPoll  = () => { if (pollRef.current) clearInterval(pollRef.current) }
  const startPoll = useCallback(() => {
    stopPoll()
    pollRef.current = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}/custom-pipeline-status`)
        setProgress(data.progress || '')
        if (data.status === 'running') return
        stopPoll()
        setRunning(false)
        if (data.status === 'success') { setResult(data.result); setError(null) }
        else if (data.status === 'error') { setError(data.error) }
      } catch {}
    }, 1200)
  }, [])

  useEffect(() => () => stopPoll(), [])

  // Başlat
  const handleRun = async () => {
    if (images.length === 0)        { setError('En az bir görsel yükleyin.'); return }
    if (selectedConfigs.size === 0) { setError('En az bir model/config seçin.'); return }

    setRunning(true); setError(null); setResult(null); setProgress('Gönderiliyor...')

    const form = new FormData()
    images.forEach(f => form.append('images', f))
    const configList = [...selectedConfigs].map(k => {
      const [engine, model_name] = k.split('/')
      return { engine, model_name }
    })
    form.append('configs', JSON.stringify(configList))

    try {
      await axios.post(`${API}/run-custom-pipeline`, form)
      startPoll()
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Karşılaştırma başlatılamadı.')
      setRunning(false)
    }
  }

  const resultRows = result ? extractMetrics(result) : []
  const docTypes   = [...new Set(resultRows.map(r => r.docType))]

  return (
    <div>
      {/* ── Başlık ── */}
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 className="section-heading" style={{ margin: 0 }}>Karşılaştırma &amp; Performans Analizi</h2>
      </div>

      {/* ── 1. Görsel Yükleme ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title" style={{ marginBottom: '1rem' }}>📁 Görsel Yükleme</div>
        <div
          className={`dropzone ${dragOver ? 'drag-over' : ''}`}
          style={{ minHeight: 100, marginBottom: images.length ? '1rem' : 0 }}
          onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <span className="dropzone-icon" style={{ fontSize: '1.5rem' }}>🖼️</span>
          <p className="dropzone-text"><strong>Sürükle &amp; bırak</strong> veya tıkla</p>
          <p className="dropzone-text" style={{ fontSize: '0.78rem', marginTop: '0.2rem' }}>
            Birden fazla görsel seçebilirsiniz · PNG, JPG, TIFF, BMP
          </p>
          <input ref={fileInputRef} type="file" accept="image/*" multiple
            style={{ display: 'none' }} onChange={e => addFiles(e.target.files)} />
        </div>

        {images.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {images.map(f => (
              <div key={f.name} style={{
                display: 'flex', alignItems: 'center', gap: '0.4rem',
                background: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                borderRadius: '6px', padding: '0.3rem 0.6rem',
                fontSize: '0.8rem', color: 'var(--text-secondary)',
              }}>
                <span>📄 {f.name}</span>
                <button onClick={() => removeImage(f.name)} style={{
                  background: 'none', border: 'none', color: 'var(--red)',
                  cursor: 'pointer', fontSize: '0.9rem', lineHeight: 1, padding: 0,
                }} title="Kaldır">✕</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── 2. Model / Config Seçimi ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
          <div className="card-title" style={{ margin: 0 }}>⚙️ Model / Konfigürasyon Seçimi</div>
          <button className="btn btn-ghost"
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
            onClick={toggleAll}>
            {allSelected ? 'Tümünü Kaldır' : 'Tümünü Seç'}
          </button>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Yeni config eklendiğinde otomatik görünür
          </span>
        </div>

        {Object.keys(engines).length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Motor listesi yükleniyor...</p>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {Object.entries(engines).flatMap(([eng, models]) =>
              models.map(m => {
                const key     = `${eng}/${m}`
                const checked = selectedConfigs.has(key)
                return (
                  <label key={key} style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    background: checked ? 'rgba(59,130,246,0.12)' : 'var(--bg-sidebar)',
                    border: `1px solid ${checked ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: '8px', padding: '0.45rem 0.85rem',
                    cursor: 'pointer', fontSize: '0.83rem',
                    color: checked ? 'var(--accent-light)' : 'var(--text-secondary)',
                    transition: 'all 0.15s', userSelect: 'none',
                  }}>
                    <input type="checkbox" checked={checked} onChange={() => toggleConfig(key)}
                      style={{ accentColor: 'var(--accent)', width: 14, height: 14 }} />
                    <span>{eng}</span>
                    <span style={{ opacity: 0.6 }}>/</span>
                    <span>{m}</span>
                  </label>
                )
              })
            )}
          </div>
        )}
      </div>

      {/* ── 3. Başlat ── */}
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '1.5rem' }}>
        <button className="btn btn-primary" onClick={handleRun} disabled={running} style={{ minWidth: 220 }}>
          {running ? `⏳ ${progress || 'İşleniyor...'}` : '▶ Karşılaştırmayı Başlat'}
        </button>
        {result && !running && (
          <span style={{ color: 'var(--green)', fontSize: '0.85rem' }}>✓ Karşılaştırma tamamlandı</span>
        )}
      </div>

      {error && <div className="error-box" style={{ marginBottom: '1.5rem' }}>⚠ {error}</div>}

      {running && (
        <div className="spinner-wrap" style={{ margin: '2rem 0' }}>
          <div className="spinner" />
          <span style={{ fontWeight: 500 }}>{progress}</span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Seçilen model × görsel kombinasyonları işleniyor...
          </span>
        </div>
      )}

      {/* ── 4. Sonuçlar ── */}
      {resultRows.length > 0 && !running && (
        <div>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', color: 'var(--text-primary)' }}>
            📊 Karşılaştırma Sonuçları
          </h3>
          {docTypes.map(dt => (
            <div key={dt} style={{ marginBottom: '2rem' }}>
              <div style={{
                fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)',
                textTransform: 'uppercase', letterSpacing: '0.08em',
                marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border)',
              }}>{dt}</div>
              <ReportCharts rows={resultRows.filter(r => r.docType === dt)} docTypeLabel={dt} />
            </div>
          ))}
        </div>
      )}

      {result && resultRows.length === 0 && !running && (
        <div className="empty-state">
          <p>Rapor oluşturuldu ancak gösterilecek metrik verisi bulunamadı.</p>
        </div>
      )}
    </div>
  )
}
