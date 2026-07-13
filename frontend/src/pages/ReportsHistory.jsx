import { useState, useEffect } from 'react'
import axios from 'axios'
import ImageViewer, { ConfidenceLegend } from '../components/ImageViewer'
import ResultsPanel from '../components/ResultsPanel'

const API = 'http://localhost:8000'

export default function ReportsHistory() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [engineFilter, setEngineFilter] = useState('')

  // Detailed Modal View
  const [selectedReport, setSelectedReport] = useState(null)
  const [detailImgSize, setDetailImgSize] = useState(null)
  const [modalLoading, setModalLoading] = useState(false)

  // Fetch reports
  const fetchReports = async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await axios.get(`${API}/past-reports`)
      setReports(data)
    } catch (err) {
      setError('Geçmiş raporlar yüklenirken bir hata oluştu.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchReports()
  }, [])

  // Delete individual report
  const handleDeleteReport = async (report) => {
    if (!window.confirm(`"${report.filename}" raporunu kalıcı olarak silmek istediğinize emin misiniz?`)) {
      return
    }

    try {
      await axios.delete(`${API}/past-reports/${report.engine}/${report.model_name}/${report.id}`)
      setReports(prev => prev.filter(r => r.id !== report.id))
      if (selectedReport?.id === report.id) {
        setSelectedReport(null)
      }
    } catch (err) {
      alert('Rapor silinirken bir hata oluştu.')
    }
  }

  // Clear all reports
  const handleClearAll = async () => {
    if (!window.confirm('Tüm geçmiş tarama raporlarını ve yüklenen görselleri temizlemek istediğinize emin misiniz? Bu işlem geri alınamaz.')) {
      return
    }

    try {
      await axios.post(`${API}/clear-web-outputs`)
      setReports([])
      setSelectedReport(null)
    } catch (err) {
      alert('Temizleme işlemi başarısız oldu.')
    }
  }

  // Open modal and compute image dimensions
  const handleOpenDetail = (report) => {
    setSelectedReport(report)
    setDetailImgSize(null)
    setModalLoading(true)

    if (report.image_url) {
      const img = new Image()
      img.onload = () => {
        setDetailImgSize({ w: img.naturalWidth, h: img.naturalHeight })
        setModalLoading(false)
      }
      img.onerror = () => {
        setModalLoading(false)
      }
      img.src = `${API}${report.image_url}`
    } else {
      setModalLoading(false)
    }
  }

  // Filter reports
  const filteredReports = reports.filter(report => {
    const filenameMatch = report.filename.toLowerCase().includes(searchQuery.toLowerCase())
    const engineMatch = engineFilter === '' || report.engine === engineFilter
    return filenameMatch && engineMatch
  })

  // Format timestamp to localized readable string
  const formatDateTime = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString('tr-TR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  // Available unique engines in the history
  const uniqueEngines = [...new Set(reports.map(r => r.engine))]

  return (
    <div>
      {/* Header controls with title and Clear All button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <h2 className="section-heading" style={{ margin: 0 }}>Geçmiş Raporlar</h2>
        {reports.length > 0 && (
          <button
            className="btn"
            onClick={handleClearAll}
            style={{ borderColor: 'var(--red)', color: 'var(--red)', fontSize: '0.85rem', padding: '0.5rem 1rem' }}
          >
            🗑️ Tüm Geçmişi Temizle
          </button>
        )}
      </div>

      {error && (
        <div className="error-box" style={{ marginBottom: '1.5rem' }}>
          ⚠️ {error}
        </div>
      )}

      {/* ── Search & Filter Controls ── */}
      <div className="card history-controls">
        <div className="form-group" style={{ flex: 1, minWidth: '200px', margin: 0 }}>
          <label className="form-label">Dosya Adı Ara</label>
          <input
            type="text"
            className="form-select"
            style={{ width: '100%', padding: '0.45rem 0.75rem' }}
            placeholder="Örn: dekont"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="form-group" style={{ width: '180px', margin: 0 }}>
          <label className="form-label">OCR Motoru</label>
          <select
            className="form-select"
            value={engineFilter}
            onChange={(e) => setEngineFilter(e.target.value)}
            style={{ width: '100%' }}
          >
            <option value="">Tümü</option>
            {uniqueEngines.map(eng => (
              <option key={eng} value={eng}>{eng}</option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Reports Grid ── */}
      {loading ? (
        <div className="spinner-wrap" style={{ margin: '4rem 0' }}>
          <div className="spinner" />
          <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Raporlar yükleniyor...</p>
        </div>
      ) : filteredReports.length === 0 ? (
        <div className="empty-state card">
          <h3>Rapor Bulunamadı</h3>
          <p>
            {reports.length === 0
              ? 'Henüz hiçbir belge taranmamış. Belge Tarama sekmesinden yeni bir analiz başlatabilirsiniz.'
              : 'Arama kriterlerinize uygun geçmiş rapor bulunamadı.'}
          </p>
        </div>
      ) : (
        <div className="history-grid">
          {filteredReports.map((report) => {
            const m = report.metrics ?? {}
            const isDriverLicense = m.doc_type === 'surucubelgesi'
            return (
              <div key={report.id} className="history-card">
                {/* Header */}
                <div className="history-card-header">
                  <div className="history-card-badges">
                    <span className="engine-badge">{report.engine}</span>
                    <span className="model-badge">{report.model_name}</span>
                  </div>
                  <div className="delete-btn-wrap">
                    <button
                      onClick={() => handleDeleteReport(report)}
                      title="Raporu sil"
                    >
                      🗑️
                    </button>
                  </div>
                </div>

                {/* Body */}
                <div className="history-card-body">
                  {report.image_url ? (
                    <img
                      src={`${API}${report.image_url}`}
                      alt={report.filename}
                      className="history-card-thumb"
                      onError={(e) => {
                        e.target.style.display = 'none'
                        e.target.nextSibling.style.display = 'flex'
                      }}
                    />
                  ) : null}
                  <div className="history-card-thumb-empty" style={{ display: report.image_url ? 'none' : 'flex' }}>
                    📄
                  </div>

                  <div className="history-card-info">
                    <div>
                      <div className="history-card-name" title={report.filename}>
                        {report.filename}
                      </div>
                      <div className="history-card-date">
                        {formatDateTime(report.timestamp)}
                      </div>
                    </div>

                    <div className="history-card-metrics">
                      {m.avg_confidence != null && (
                        <div className="history-metric">
                          Güven: <strong>{m.avg_confidence.toFixed(1)}%</strong>
                        </div>
                      )}
                      {isDriverLicense && m.avg_cer != null && (
                        <div className="history-metric">
                          CER: <strong>{(m.avg_cer * 100).toFixed(1)}%</strong>
                        </div>
                      )}
                      {m.avg_total_time_seconds != null && (
                        <div className="history-metric">
                          Süre: <strong>{m.avg_total_time_seconds.toFixed(2)}s</strong>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Footer / Action */}
                <div className="history-card-footer">
                  <button
                    className="btn btn-ghost"
                    onClick={() => handleOpenDetail(report)}
                    style={{ fontSize: '0.8rem', padding: '0.3rem 0.75rem' }}
                  >
                    Detayları İncele →
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Detailed Modal ── */}
      {selectedReport && (
        <div className="modal-overlay" onClick={() => setSelectedReport(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="modal-header">
              <div className="modal-header-title">
                <h3>{selectedReport.filename}</h3>
                <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <span>OCR Motoru: <strong>{selectedReport.engine} ({selectedReport.model_name})</strong></span>
                  <span>•</span>
                  <span>Tarih: <strong>{formatDateTime(selectedReport.timestamp)}</strong></span>
                </div>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedReport(null)}>
                ×
              </button>
            </div>

            {/* Modal Body */}
            <div className="modal-body">
              {modalLoading ? (
                <div className="spinner-wrap" style={{ margin: '4rem 0' }}>
                  <div className="spinner" />
                  <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Görsel ve detaylar hazırlanıyor...</p>
                </div>
              ) : (
                <div className="scanner-layout">
                  {/* Left: Image Viewer with boxes */}
                  <div>
                    <div style={{ marginBottom: '0.75rem' }}>
                      <ConfidenceLegend />
                    </div>
                    {selectedReport.image_url ? (
                      <ImageViewer
                        imgSrc={`${API}${selectedReport.image_url}`}
                        words={selectedReport.data?.words ?? []}
                        imgSize={detailImgSize}
                      />
                    ) : (
                      <div className="empty-state card">
                        <h3>Görsel Bulunamadı</h3>
                        <p>Bu rapor için yüklenen orijinal görsel diskte bulunamadı.</p>
                      </div>
                    )}
                  </div>

                  {/* Right: Results Panel */}
                  <div>
                    <ResultsPanel data={selectedReport.data} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
