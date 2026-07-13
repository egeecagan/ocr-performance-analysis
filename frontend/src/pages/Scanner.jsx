/**
 * Scanner.jsx — Belge Tarama Paneli
 *
 * 1. Gorsel yukle (drag-and-drop veya dosya sec)
 * 2. OCR motoru ve model versiyonu sec
 * 3. "Belgeyi Isle" butonuna bas
 * 4. Sol: gorsel + bounding box  |  Sag: metrikler + alan tablosu
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import ImageViewer, { ConfidenceLegend } from '../components/ImageViewer'
import ResultsPanel from '../components/ResultsPanel'

const API = 'http://localhost:8000'

export default function Scanner() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [engines, setEngines] = useState({})          // {engine: [model,...]}
  const [engine, setEngine] = useState('')
  const [modelName, setModelName] = useState('')
  const [file, setFile] = useState(null)        // File object
  const [previewUrl, setPreviewUrl] = useState(null)        // object URL
  const [imgSize, setImgSize] = useState(null)        // {w, h} piksel
  const [result, setResult] = useState(null)        // API yaniti
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const [clearStatus, setClearStatus] = useState(null)

  const fileInputRef = useRef(null)

  const handleClearWebOutputs = async () => {
    try {
      await axios.post(`${API}/clear-web-outputs`)
      setClearStatus('success')
      setTimeout(() => setClearStatus(null), 3000)
    } catch (err) {
      setClearStatus('error')
      setTimeout(() => setClearStatus(null), 3000)
    }
  }

  // ── Motor listesini yukle ─────────────────────────────────────────────────
  useEffect(() => {
    axios.get(`${API}/engines`)
      .then(r => {
        setEngines(r.data)
        const first = Object.keys(r.data)[0] ?? ''
        setEngine(first)
        setModelName(r.data[first]?.[0] ?? '')
      })
      .catch(() => setError('API\'ye ulasilamadi. Uvicorn sunucusunun calistigini kontrol edin.'))
  }, [])

  // Engine degisince ilk model versiyonunu sec
  useEffect(() => {
    if (engine && engines[engine]) setModelName(engines[engine][0])
  }, [engine, engines])

  // ── Gorsel secilince on izleme olustur ───────────────────────────────────
  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreviewUrl(url)
    // Gorselin piksel boyutunu al
    const img = new Image()
    img.onload = () => setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
  }, [])

  // ── Drag-and-drop ─────────────────────────────────────────────────────────
  const onDragOver = (e) => { e.preventDefault(); setDragOver(true) }
  const onDragLeave = () => setDragOver(false)
  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  // ── OCR islemini gonder ───────────────────────────────────────────────────
  const handleProcess = async () => {
    if (!file || !engine || !modelName) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('engine', engine)
      form.append('model_name', modelName)
      const { data } = await axios.post(`${API}/process`, form)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Bilinmeyen bir hata olustu.')
    } finally {
      setLoading(false)
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h2 className="section-heading" style={{ margin: 0 }}>Belge Tarama ve Analiz</h2>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            className="btn btn-ghost"
            onClick={handleClearWebOutputs}
            style={{ borderColor: 'var(--red)', color: 'var(--red)', fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
          >
            🗑️ Tüm Web Çıktılarını Temizle
          </button>
          {clearStatus === 'success' && <span style={{ color: 'var(--green)', fontSize: '0.85rem' }}>✓ Temizlendi</span>}
          {clearStatus === 'error' && <span style={{ color: 'var(--red)', fontSize: '0.85rem' }}>✗ Hata oluştu</span>}
        </div>
      </div>

      {/* ── Model seçim kontrolü ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>

          {/* Engine dropdown */}
          <div className="form-group" style={{ minWidth: '160px' }}>
            <label className="form-label">OCR Motoru</label>
            <select
              className="form-select"
              value={engine}
              onChange={e => setEngine(e.target.value)}
            >
              {Object.keys(engines).map(eng => (
                <option key={eng} value={eng}>{eng}</option>
              ))}
            </select>
          </div>

          {/* Model versiyonu dropdown */}
          <div className="form-group" style={{ minWidth: '180px' }}>
            <label className="form-label">Konfigürasyon</label>
            <select
              className="form-select"
              value={modelName}
              onChange={e => setModelName(e.target.value)}
            >
              {(engines[engine] ?? []).map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* İşle butonu */}
          <button
            className="btn btn-primary"
            style={{ marginLeft: 'auto', minWidth: '170px' }}
            onClick={handleProcess}
            disabled={!file || loading}
          >
            {loading ? '⏳ İşleniyor...' : '▶ Belgeyi İşle'}
          </button>
        </div>
      </div>

      {/* ── Hata kutusu ── */}
      {error && <div className="error-box" style={{ marginBottom: '1.5rem' }}>⚠ {error}</div>}

      {/* ── Ana içerik: sol görsel | sağ sonuçlar ── */}
      <div className="scanner-layout">

        {/* ── Sol panel: Dropzone veya Görsel + BBox ── */}
        <div>
          {/* Confidence legend (her zaman görünür) */}
          <div style={{ marginBottom: '0.75rem' }}>
            <ConfidenceLegend />
          </div>

          {/* Dropzone */}
          {!previewUrl ? (
            <div
              className={`dropzone ${dragOver ? 'drag-over' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="dropzone-icon">📄</span>
              <p className="dropzone-text">
                <strong>Sürükle & bırak</strong> veya tıkla
              </p>
              <p className="dropzone-text" style={{ marginTop: '0.25rem', fontSize: '0.78rem' }}>
                PNG, JPG, TIFF, BMP desteklenir
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={e => handleFile(e.target.files[0])}
              />
            </div>
          ) : (
            <div className="card" style={{ padding: '0' }}>
              {/* Görsel başlığı ve değiştir butonu */}
              <div style={{
                display: 'flex', alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.75rem 1rem',
                borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  📎 {file?.name}
                </span>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: '0.75rem', padding: '0.3rem 0.75rem' }}
                  onClick={() => {
                    setFile(null)
                    setPreviewUrl(null)
                    setResult(null)
                    setImgSize(null)
                  }}
                >
                  Değiştir
                </button>
              </div>
              {/* Görsel + bounding box'lar */}
              <div style={{ padding: '0.5rem' }}>
                <ImageViewer
                  imgSrc={previewUrl}
                  words={loading ? [] : (result?.words ?? [])}
                  imgSize={imgSize}
                />
              </div>
            </div>
          )}
        </div>

        {/* ── Sağ panel: Yükleme animasyonu veya Sonuçlar ── */}
        <div>
          {loading ? (
            <div className="spinner-wrap">
              <div className="spinner" />
              <span>OCR işleniyor...</span>
            </div>
          ) : (
            <ResultsPanel data={result} />
          )}
        </div>
      </div>
    </div>
  )
}
