/**
 * ResultsPanel.jsx
 *
 * OCR sonuclarini gosteren sag panel:
 *   - Metrik KPI kartlari (sure, kelime sayisi, confidence, CER, WER)
 *   - Alan eslesmesi tablosu (field_results)
 *   - Ortak alan tablosu (common_field_results)
 */

function MetricCard({ label, value, unit = '', color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={color ? { color } : {}}>
        {value ?? '—'}
        {unit && <span className="kpi-unit">{unit}</span>}
      </div>
    </div>
  )
}

function FieldTable({ title, rows }) {
  if (!rows || rows.length === 0) return null
  return (
    <div className="card" style={{ marginBottom: '1rem' }}>
      <div className="card-title">{title}</div>
      <table className="field-table">
        <thead>
          <tr>
            <th>Alan</th>
            <th>Bulunan Değer</th>
            <th>CER</th>
            <th>Durum</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([field, data], i) => (
            <tr key={i}>
              <td>{field}</td>
              <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                {data.matched_substring ?? <em style={{ opacity: 0.4 }}>—</em>}
              </td>
              <td>
                {data.cer != null ? (
                  <span style={{
                    color: data.cer < 0.2 ? 'var(--green)' :
                           data.cer < 0.5 ? 'var(--yellow)' : 'var(--red)'
                  }}>
                    {data.cer.toFixed(3)}
                  </span>
                ) : '—'}
              </td>
              <td>
                <span className={`badge ${data.found ? 'found' : 'missing'}`}>
                  {data.found ? '✓ Bulundu' : '✗ Bulunamadı'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function ResultsPanel({ data }) {
  if (!data) {
    return (
      <div className="empty-state">
        <h3>Sonuç Bekleniyor</h3>
        <p>Bir görsel yükleyip "Belgeyi İşle" butonuna basın.</p>
      </div>
    )
  }

  const m = data.metrics ?? {}
  const cf = m.common_fields ?? {}

  const fieldRows = data.field_results
    ? Object.entries(data.field_results)
    : []

  const commonRows = data.common_field_results
    ? Object.entries(data.common_field_results)
    : []

  return (
    <div>
      {/* ── KPI Kartları (Ana Metrikler) ── */}
      <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
        Genel Metrikler ({m.doc_type === 'surucubelgesi' ? 'Sürücü Belgesi' : 'Dekont'})
      </h3>
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: '1.5rem' }}>
        <MetricCard
          label="Dosya Sayısı"
          value={m.file_count}
          unit="adet"
        />
        <MetricCard
          label="Ort. İşlem Süresi"
          value={m.avg_total_time_seconds?.toFixed(4)}
          unit="s"
        />
        <MetricCard
          label="Ortalama Güven"
          value={m.avg_confidence != null ? m.avg_confidence.toFixed(2) : null}
          unit="%"
          color={m.avg_confidence >= 80 ? 'var(--green)' : m.avg_confidence >= 50 ? 'var(--yellow)' : 'var(--red)'}
        />
        {m.doc_type === 'surucubelgesi' && (
          <MetricCard
            label="Geçerli Kelime Sayısı"
            value={m.valid_word_count}
            unit="adet"
          />
        )}
      </div>

      {/* ── Sürücü Belgesine Özel Metrikler ── */}
      {m.doc_type === 'surucubelgesi' && (
        <>
          <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
            Sürücü Belgesi Özel Metrikleri
          </h3>
          <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: '1.5rem' }}>
            <MetricCard
              label="Ortalama CER"
              value={m.avg_cer != null ? (m.avg_cer * 100).toFixed(2) : null}
              unit="%"
              color={m.avg_cer < 0.1 ? 'var(--green)' : m.avg_cer < 0.3 ? 'var(--yellow)' : 'var(--red)'}
            />
            <MetricCard
              label="Ortalama WER"
              value={m.avg_wer != null ? (m.avg_wer * 100).toFixed(2) : null}
              unit="%"
              color={m.avg_wer < 0.15 ? 'var(--green)' : m.avg_wer < 0.4 ? 'var(--yellow)' : 'var(--red)'}
            />
            <MetricCard
              label="Alan Eşleşme Oranı"
              value={m.avg_field_match_ratio != null ? (m.avg_field_match_ratio * 100).toFixed(1) : null}
              unit="%"
            />
            <MetricCard
              label="Doğru Eşleşme Oranı"
              value={m.is_match_true_ratio != null ? (m.is_match_true_ratio * 100).toFixed(1) : null}
              unit="%"
            />
          </div>
        </>
      )}

      {/* ── Ortak Alanlar (Common Fields) Metrikleri ── */}
      <h3 style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
        Ortak Alan (Common Fields) Metrikleri
      </h3>
      <div className="kpi-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: '1.5rem' }}>
        <MetricCard
          label="Ortak Alan CER"
          value={cf.avg_cer != null ? (cf.avg_cer * 100).toFixed(2) : null}
          unit="%"
          color={cf.avg_cer < 0.1 ? 'var(--green)' : cf.avg_cer < 0.3 ? 'var(--yellow)' : 'var(--red)'}
        />
        <MetricCard
          label="Ortak Alan WER"
          value={cf.avg_wer != null ? (cf.avg_wer * 100).toFixed(2) : null}
          unit="%"
          color={cf.avg_wer < 0.15 ? 'var(--green)' : cf.avg_wer < 0.4 ? 'var(--yellow)' : 'var(--red)'}
        />
        <MetricCard
          label="Alan Eşleşme Oranı"
          value={cf.avg_common_field_match_ratio != null ? (cf.avg_common_field_match_ratio * 100).toFixed(1) : null}
          unit="%"
        />
        <MetricCard
          label="Alan Ort. Güven"
          value={cf.avg_common_field_confidence != null ? cf.avg_common_field_confidence.toFixed(2) : null}
          unit="%"
        />
        <MetricCard
          label="Bulunma Oranı"
          value={cf.found_true_ratio != null ? (cf.found_true_ratio * 100).toFixed(1) : null}
          unit="%"
        />
      </div>

      {/* ── Alan Tabloları ── */}
      <FieldTable title="Ground Truth Alan Eşleşmesi" rows={fieldRows} />
      <FieldTable title="Ortak Alan Kontrolü" rows={commonRows} />

      {/* Ground truth yoksa bilgi notu */}
      {!data.has_ground_truth && (
        <div style={{
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          padding: '0.75rem 1rem',
          border: '1px solid var(--border)',
          borderRadius: '8px',
          marginTop: '0.5rem',
        }}>
          ℹ️ Bu görsel için ground truth YAML dosyası bulunamadı.
          Alan eşleşme metrikleri hesaplanmadı.
        </div>
      )}
    </div>
  )
}

