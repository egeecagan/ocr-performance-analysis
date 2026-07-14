/**
 * Scanner.jsx — Belge Tarama Paneli
 *
 * 1. Gorsel yukle (drag-and-drop veya dosya sec)
 * 2. OCR motoru ve model versiyonu sec (veya yeni konfigürasyon ekle)
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

  // Yeni Konfigürasyon Modal State'leri
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [newConfigSchema, setNewConfigSchema] = useState(null)
  const [newConfigName, setNewConfigName]     = useState('')
  const [newConfigEngineSettings, setNewConfigEngineSettings] = useState({})
  const [newConfigPreprocessing, setNewConfigPreprocessing]   = useState({})
  const [configSubmitError, setConfigSubmitError]             = useState(null)
  const [configSubmitSuccess, setConfigSubmitSuccess]         = useState(false)

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
  const loadEngines = (selectNewConfig = null) => {
    axios.get(`${API}/engines`)
      .then(r => {
        setEngines(r.data)
        if (selectNewConfig) {
          setEngine(selectNewConfig.engine)
          setModelName(selectNewConfig.modelName)
        } else {
          // Varsayılan seçim
          const currentEngines = Object.keys(r.data)
          if (currentEngines.length > 0) {
            if (!engine || !r.data[engine]) {
              const first = currentEngines[0]
              setEngine(first)
              setModelName(r.data[first]?.[0] ?? '')
            }
          }
        }
      })
      .catch(() => setError('API\'ye ulasilamadi. Uvicorn sunucusunun calistigini kontrol edin.'))
  }

  useEffect(() => {
    loadEngines()
  }, [])

  // Engine degisince ilk model versiyonunu sec (Eğer konfigürasyon eklemeden tetiklenmemişse)
  useEffect(() => {
    if (engine && engines[engine] && !showConfigModal) {
      // Eğer seçili modelName mevcut engine listesinde yoksa ilkini seç
      if (!engines[engine].includes(modelName)) {
        setModelName(engines[engine][0] ?? '')
      }
    }
  }, [engine, engines, showConfigModal])

  // Yeni konfigürasyon için şemayı çek
  useEffect(() => {
    if (showConfigModal && engine) {
      axios.get(`${API}/config-schema/${engine}`)
        .then(r => {
          setNewConfigSchema(r.data)
          
          // Varsayılan ayarları yükle
          const initialSettings = {}
          if (r.data.engine_settings) {
            Object.values(r.data.engine_settings).flat().forEach(field => {
              initialSettings[field.key] = field.default
            })
          }
          
          const initialPrep = {}
          if (r.data.preprocessing) {
            r.data.preprocessing.forEach(step => {
              initialPrep[step.key] = step.default
              if (step.params) {
                step.params.forEach(p => {
                  initialPrep[p.key] = p.default
                })
              }
            })
          }

          setNewConfigEngineSettings(initialSettings)
          setNewConfigPreprocessing(initialPrep)
          setNewConfigName('')
          setConfigSubmitError(null)
          setConfigSubmitSuccess(false)
        })
        .catch(() => {
          setConfigSubmitError('Konfigürasyon şeması yüklenemedi.')
        })
    }
  }, [engine, showConfigModal])

  // ── Gorsel secilince on izleme olustur ───────────────────────────────────
  const handleFile = useCallback((f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreviewUrl(url)
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

  // Yeni Konfigürasyon Kaydet
  const handleSaveConfig = async (e) => {
    e.preventDefault()
    if (!newConfigName.trim()) {
      setConfigSubmitError('Lütfen geçerli bir konfigürasyon adı girin.')
      return
    }
    setConfigSubmitError(null)
    try {
      await axios.post(`${API}/configs/${engine}`, {
        config_name: newConfigName,
        engine_settings: newConfigEngineSettings,
        preprocessing: newConfigPreprocessing
      })
      setConfigSubmitSuccess(true)
      
      // Motor listesini güncelle ve kaydettiğimiz config'i otomatik seç
      loadEngines({ engine: engine, modelName: newConfigName })
      
      setTimeout(() => {
        setShowConfigModal(false)
      }, 1000)
    } catch (err) {
      setConfigSubmitError(err.response?.data?.detail ?? 'Konfigürasyon kaydedilemedi.')
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
            Tüm Web Çıktılarını Temizle
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

          {/* Model versiyonu dropdown ve Ekle butonu */}
          <div className="form-group" style={{ minWidth: '220px' }}>
            <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Konfigürasyon</span>
              <button
                type="button"
                onClick={() => setShowConfigModal(true)}
                style={{
                  background: 'none', border: 'none', color: 'var(--accent-light)',
                  cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600, padding: 0
                }}
              >
                ➕ Yeni Ekle
              </button>
            </label>
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

      {/* ── 5. KONFİGÜRASYON EKLE MODAL ── */}
      {showConfigModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex',
          alignItems: 'center', justifyContent: 'center', zIndex: 1000,
          backdropFilter: 'blur(4px)'
        }}>
          <div style={{
            backgroundColor: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '12px', padding: '2rem', width: '90%', maxWidth: '680px',
            maxHeight: '85vh', overflowY: 'auto', boxShadow: 'var(--shadow-lg)'
          }}>
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.8rem' }}>
              <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-primary)' }}>➕ Yeni Konfigürasyon Oluştur ({engine.toUpperCase()})</h3>
              <button
                onClick={() => setShowConfigModal(false)}
                style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', fontSize: '1.4rem', cursor: 'pointer' }}
              >✕</button>
            </div>

            {/* Modal Body / Form */}
            <form onSubmit={handleSaveConfig}>
              {/* Konfigürasyon Adı */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-secondary)' }}>Konfigürasyon Adı</label>
                <input
                  type="text"
                  placeholder="Örnek: model_aydinlatma_v1"
                  value={newConfigName}
                  onChange={(e) => setNewConfigName(e.target.value)}
                  style={{
                    width: '100%', padding: '0.6rem', borderRadius: '8px',
                    backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                    color: 'var(--text-primary)', fontSize: '0.9rem'
                  }}
                  required
                />
              </div>

              {newConfigSchema ? (
                <div>
                  {/* Engine-Specific Model Ayarları */}
                  {Object.keys(newConfigSchema.engine_settings || {}).length > 0 && (
                    <div style={{ marginBottom: '1.5rem' }}>
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 600, borderBottom: '1px solid var(--border)', paddingBottom: '0.4rem', marginBottom: '0.8rem', color: 'var(--accent-light)' }}>
                        ⚙️ Model Ayarları (Model Settings)
                      </h4>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {Object.entries(newConfigSchema.engine_settings).flatMap(([blockName, fields]) =>
                          fields.map(field => (
                            <div key={field.key} style={{ gridColumn: field.type === 'str' ? 'span 2' : 'span 1' }}>
                              <label style={{ display: 'block', fontSize: '0.78rem', marginBottom: '0.3rem', color: 'var(--text-secondary)' }}>
                                {field.label}
                              </label>
                              
                              {field.type === 'bool' && (
                                <input
                                  type="checkbox"
                                  checked={!!newConfigEngineSettings[field.key]}
                                  onChange={(e) => setNewConfigEngineSettings(prev => ({ ...prev, [field.key]: e.target.checked ? 1 : 0 }))}
                                  style={{ width: 16, height: 16, accentColor: 'var(--accent)' }}
                                />
                              )}

                              {field.type === 'select' && (
                                <select
                                  value={newConfigEngineSettings[field.key] ?? field.default}
                                  onChange={(e) => setNewConfigEngineSettings(prev => ({ ...prev, [field.key]: field.default === true || field.default === false ? e.target.value === 'true' : isNaN(e.target.value) ? e.target.value : Number(e.target.value) }))}
                                  style={{
                                    width: '100%', padding: '0.45rem', borderRadius: '6px',
                                    backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                    color: 'var(--text-primary)', fontSize: '0.82rem'
                                  }}
                                >
                                  {field.options.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.desc}</option>
                                  ))}
                                </select>
                              )}

                              {field.type === 'int' && (
                                <input
                                  type="number"
                                  value={newConfigEngineSettings[field.key] ?? field.default}
                                  onChange={(e) => setNewConfigEngineSettings(prev => ({ ...prev, [field.key]: parseInt(e.target.value) || 0 }))}
                                  style={{
                                    width: '100%', padding: '0.45rem', borderRadius: '6px',
                                    backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                    color: 'var(--text-primary)', fontSize: '0.82rem'
                                  }}
                                />
                              )}

                              {field.type === 'float' && (
                                <input
                                  type="number"
                                  step="0.05"
                                  value={newConfigEngineSettings[field.key] ?? field.default}
                                  onChange={(e) => setNewConfigEngineSettings(prev => ({ ...prev, [field.key]: parseFloat(e.target.value) || 0 }))}
                                  style={{
                                    width: '100%', padding: '0.45rem', borderRadius: '6px',
                                    backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                    color: 'var(--text-primary)', fontSize: '0.82rem'
                                  }}
                                />
                              )}

                              {field.type === 'str' && (
                                <input
                                  type="text"
                                  value={newConfigEngineSettings[field.key] ?? field.default}
                                  onChange={(e) => setNewConfigEngineSettings(prev => ({ ...prev, [field.key]: e.target.value }))}
                                  style={{
                                    width: '100%', padding: '0.45rem', borderRadius: '6px',
                                    backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                    color: 'var(--text-primary)', fontSize: '0.82rem'
                                  }}
                                />
                              )}
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  )}

                  {/* Görüntü Ön İşleme (Preprocessing) Ayarları */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.95rem', fontWeight: 600, borderBottom: '1px solid var(--border)', paddingBottom: '0.4rem', marginBottom: '0.8rem', color: 'var(--accent-light)' }}>
                      🖼️ Görüntü Ön İşleme (Preprocessing Settings)
                    </h4>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                      {newConfigSchema.preprocessing.map(step => {
                        const isEnabled = !!newConfigPreprocessing[step.key]
                        return (
                          <div key={step.key} style={{
                            border: '1px solid var(--border)', borderRadius: '8px', padding: '0.6rem 0.8rem',
                            backgroundColor: isEnabled ? 'rgba(59,130,246,0.04)' : 'transparent'
                          }}>
                            {/* Toggle Switch */}
                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', userSelect: 'none', margin: 0 }}>
                              <input
                                type="checkbox"
                                checked={isEnabled}
                                onChange={(e) => setNewConfigPreprocessing(prev => ({ ...prev, [step.key]: e.target.checked }))}
                                style={{ width: 16, height: 16, accentColor: 'var(--accent)' }}
                              />
                              <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)' }}>{step.label}</span>
                            </label>

                            {/* Ek Parametreler */}
                            {isEnabled && step.params && (
                              <div style={{
                                marginTop: '0.5rem', borderTop: '1px dashed var(--border)', paddingTop: '0.5rem',
                                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem'
                              }}>
                                {step.params.map(p => (
                                  <div key={p.key}>
                                    <label style={{ display: 'block', fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>{p.label}</label>
                                    {p.type === 'select' ? (
                                      <select
                                        value={newConfigPreprocessing[p.key] ?? p.default}
                                        onChange={(e) => setNewConfigPreprocessing(prev => ({ ...prev, [p.key]: e.target.value }))}
                                        style={{
                                          width: '100%', padding: '0.35rem', borderRadius: '4px',
                                          backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                          color: 'var(--text-primary)', fontSize: '0.78rem'
                                        }}
                                      >
                                        {p.options.map(opt => (
                                          <option key={opt.value} value={opt.value}>{opt.desc}</option>
                                        ))}
                                      </select>
                                    ) : (
                                      <input
                                        type="number"
                                        step={p.type === 'float' ? '0.05' : '1'}
                                        value={newConfigPreprocessing[p.key] ?? p.default}
                                        onChange={(e) => setNewConfigPreprocessing(prev => ({ ...prev, [p.key]: p.type === 'float' ? parseFloat(e.target.value) || 0 : parseInt(e.target.value) || 0 }))}
                                        style={{
                                          width: '100%', padding: '0.35rem', borderRadius: '4px',
                                          backgroundColor: 'var(--bg-sidebar)', border: '1px solid var(--border)',
                                          color: 'var(--text-primary)', fontSize: '0.78rem'
                                        }}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="spinner-wrap" style={{ margin: '2rem 0' }}>
                  <div className="spinner" />
                  <span>Parametreler yükleniyor...</span>
                </div>
              )}

              {/* Status / Errors */}
              {configSubmitError && (
                <div className="error-box" style={{ marginBottom: '1rem' }}>⚠ {configSubmitError}</div>
              )}

              {configSubmitSuccess && (
                <div className="success-box" style={{ marginBottom: '1rem', color: 'var(--green)', fontSize: '0.85rem' }}>
                  ✓ Konfigürasyon başarıyla oluşturuldu ve kaydedildi!
                </div>
              )}

              {/* Modal Footer / Save */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.8rem', borderTop: '1px solid var(--border)', paddingTop: '1rem', marginTop: '1rem' }}>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => setShowConfigModal(false)}
                  style={{ fontSize: '0.85rem' }}
                >İptal</button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={configSubmitSuccess}
                  style={{ fontSize: '0.85rem', minWidth: '150px' }}
                >Kaydet</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
