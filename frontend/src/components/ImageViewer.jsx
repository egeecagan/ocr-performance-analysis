/**
 * ImageViewer.jsx
 *
 * Yuklenen gorsel uzerine OCR kelimelerinin bounding box'larini cizer.
 * Fare kutucugun uzerine geldiginde tooltip gosterir.
 *
 * Props:
 *   imgSrc  : string   — gorselin object URL'i (URL.createObjectURL)
 *   words   : array    — OCR'in donduqrugu kelime listesi
 *   imgSize : {w, h}   — gorselin orijinal piksel boyutu
 */

import { useRef, useState, useEffect } from 'react'

// ── Renk esikleri ──────────────────────────────────────────────────────────
const CONF_COLORS = {
  green:  { min: 80,  max: 100, stroke: '#22c55e', fill: 'rgba(34,197,94,0.10)' },
  yellow: { min: 50,  max: 79,  stroke: '#eab308', fill: 'rgba(234,179, 8,0.10)' },
  red:    { min: 0,   max: 49,  stroke: '#ef4444', fill: 'rgba(239, 68,68,0.10)' },
}

function confColor(conf) {
  if (conf >= 80) return CONF_COLORS.green
  if (conf >= 50) return CONF_COLORS.yellow
  return CONF_COLORS.red
}

// ── Yardimci: bbox dizisi [x1,y1,x2,y2] veya [[x,y],...] -> {x,y,w,h} ────
function normalizeBbox(bbox) {
  if (!bbox) return null
  // Duz dizi [x1,y1,x2,y2]
  if (typeof bbox[0] === 'number') {
    const [x1, y1, x2, y2] = bbox
    return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
  }
  // [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
  if (Array.isArray(bbox[0])) {
    const xs = bbox.map(p => p[0])
    const ys = bbox.map(p => p[1])
    const x1 = Math.min(...xs), x2 = Math.max(...xs)
    const y1 = Math.min(...ys), y2 = Math.max(...ys)
    return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
  }
  return null
}

// ── Confidence Legend ──────────────────────────────────────────────────────
export function ConfidenceLegend() {
  return (
    <div className="conf-legend">
      <span className="conf-legend-title">Güven Renk Kodu</span>
      <span className="legend-item">
        <span className="legend-dot green" />
        <span>80 – 100 &nbsp;Yüksek</span>
      </span>
      <span className="legend-item">
        <span className="legend-dot yellow" />
        <span>50 – 79 &nbsp;Orta</span>
      </span>
      <span className="legend-item">
        <span className="legend-dot red" />
        <span>0 – 49 &nbsp;Düşük</span>
      </span>
    </div>
  )
}

// ── Ana bileşen ────────────────────────────────────────────────────────────
export default function ImageViewer({ imgSrc, words = [], imgSize }) {
  const containerRef = useRef(null)
  const [tooltip, setTooltip]   = useState(null)   // { x, y, word }
  const [scale, setScale]       = useState({ sx: 1, sy: 1 })

  // Gorsel boyutu degistiginde olcek faktoru hesapla
  useEffect(() => {
    if (!containerRef.current || !imgSize?.w || !imgSize?.h) return
    const ro = new ResizeObserver(() => {
      const rect = containerRef.current.getBoundingClientRect()
      setScale({ sx: rect.width / imgSize.w, sy: rect.height / imgSize.h })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [imgSize])

  return (
    <div>
      {/* Gorusel + kutucuklar */}
      <div className="viewer-wrapper" ref={containerRef}>
        <img src={imgSrc} alt="OCR gorseli" draggable={false} />

        {/* SVG katmani — pointer events aktif kutucuklar icin */}
        <div className="bbox-overlay" style={{ pointerEvents: 'none' }}>
          <svg style={{ pointerEvents: 'all' }}>
            {words.map((word, i) => {
              const bbox = normalizeBbox(word.bbox)
              if (!bbox) return null
              const { stroke, fill } = confColor(word.confidence ?? 0)
              const x = bbox.x * scale.sx
              const y = bbox.y * scale.sy
              const w = bbox.w * scale.sx
              const h = bbox.h * scale.sy
              return (
                <rect
                  key={i}
                  x={x} y={y} width={w} height={h}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={1.5}
                  rx={2}
                  style={{ cursor: 'crosshair', pointerEvents: 'all' }}
                  onMouseEnter={(e) => setTooltip({ x: e.clientX, y: e.clientY, word })}
                  onMouseMove={(e)  => setTooltip(t => ({ ...t, x: e.clientX, y: e.clientY }))}
                  onMouseLeave={()  => setTooltip(null)}
                />
              )
            })}
          </svg>
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="bbox-tooltip"
          style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
        >
          <div><strong>"{tooltip.word.text}"</strong></div>
          <div>Güven: <strong style={{ color: confColor(tooltip.word.confidence ?? 0).stroke }}>
            {(tooltip.word.confidence ?? 0).toFixed(1)}%
          </strong></div>
          {tooltip.word.cer != null && (
            <div>CER: {tooltip.word.cer.toFixed(4)}</div>
          )}
          {tooltip.word.matched_field && (
            <div style={{ color: '#60a5fa' }}>Alan: {tooltip.word.matched_field}</div>
          )}
        </div>
      )}
    </div>
  )
}
