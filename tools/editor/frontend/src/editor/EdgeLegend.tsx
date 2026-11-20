// エッジ型の凡例(§5)。信号種の色とビット幅→太さの意味を小さく示す。
// LodIndicatorの近く(キャンバス右上)に常時表示する。

export function EdgeLegend() {
  return (
    <div className="edge-legend">
      <span className="edge-legend-title">エッジ</span>
      <span className="edge-legend-item">
        <i style={{ background: 'var(--edge-default)' }} />
        論理
      </span>
      <span className="edge-legend-item">
        <i style={{ background: 'var(--edge-real)' }} />
        実数域
      </span>
      <span className="edge-legend-item">太さ=ビット幅</span>
    </div>
  );
}
