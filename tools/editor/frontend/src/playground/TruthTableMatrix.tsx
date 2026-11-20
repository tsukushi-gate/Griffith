// 真理値表検証テーブル。入力幅≤10bitのとき2^n全行を列挙し、
// 現在の回路出力(hard)と期待値(クリック編集)の一致を行別に表示する。
// 行クリックでその入力パターンを入力トグルへピン留めする。

import { useGraphStore } from '../store/graphStore';
import { usePlaygroundStore } from './playgroundStore';
import { inputBitsToPattern, matchInfo, TRUTH_TABLE_MAX_BITS } from './truthTable';

/** 一致率推移のスパークライン(学習で正解セルが増えていく様子を可視化) */
function MatchTrend({ history }: { history: number[] }) {
  if (history.length < 2) return null;
  const w = 64;
  const h = 14;
  const pts = history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * w;
      const y = h - v * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={w} height={h} className="overflow-visible" aria-hidden>
      <polyline
        points={pts}
        fill="none"
        stroke="var(--status-ok)"
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function TruthTableMatrix() {
  const ir = useGraphStore((s) => s.ir);
  const rows = usePlaygroundStore((s) => s.tableRows);
  const expected = usePlaygroundStore((s) => s.expectedTable);
  const inputBits = usePlaygroundStore((s) => s.inputBits);
  const matchHistory = usePlaygroundStore((s) => s.matchHistory);
  const traceK = usePlaygroundStore((s) => s.traceK);
  const pinPattern = usePlaygroundStore((s) => s.pinPattern);
  const startTrace = usePlaygroundStore((s) => s.startTrace);
  const toggleExpectedBit = usePlaygroundStore((s) => s.toggleExpectedBit);
  const resetExpected = usePlaygroundStore((s) => s.resetExpected);

  if (!ir) return null;
  if (!rows) {
    return (
      <div className="p-3 text-[11px] text-muted-foreground">
        入力幅が{TRUTH_TABLE_MAX_BITS}bitを超えるため真理値表の全列挙は未対応です
      </div>
    );
  }

  const info = matchInfo(rows, expected);
  const currentK = inputBitsToPattern(ir, inputBits);
  const outWidth = rows[0]?.outputBits.length ?? 0;

  return (
    <div className="flex h-full min-h-0 flex-col gap-1">
      <div className="flex shrink-0 items-center justify-between pr-2">
        <div className="text-[11px]">
          一致率{' '}
          <span
            className="font-mono font-semibold"
            style={{
              color:
                info.matched === info.total ? 'var(--status-ok)' : 'var(--status-warn)',
            }}
          >
            {info.matched}/{info.total}
          </span>
          <span className="ml-1 text-muted-foreground">
            ({info.total > 0 ? Math.round((info.matched / info.total) * 100) : 0}%)
          </span>
          <span className="ml-2 inline-flex align-middle">
            <MatchTrend history={matchHistory} />
          </span>
        </div>
        <button
          type="button"
          className="cursor-pointer rounded-md border px-2 py-0.5 text-[10px] text-muted-foreground hover:bg-muted"
          title="期待値を現在の回路出力のコピーでリセット"
          onClick={resetExpected}
        >
          期待値をリセット
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto rounded-md border">
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead className="sticky top-0 bg-[var(--panel-bg)] text-muted-foreground">
            <tr className="border-b text-left">
              <th className="px-2 py-1 font-normal">入力</th>
              <th className="px-2 py-1 font-normal">出力</th>
              <th className="px-2 py-1 font-normal">期待値(クリック編集)</th>
              <th className="px-2 py-1 text-center font-normal">一致</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isPinned = row.k === currentK;
              const isMismatch = !info.perRow[i];
              const isTraced = row.k === traceK;
              const exp = expected?.[row.k];
              return (
                <tr
                  key={row.k}
                  className={`cursor-pointer border-b last:border-b-0 hover:bg-muted/60 ${
                    isTraced
                      ? 'bg-destructive/10 outline outline-1 outline-[var(--edge-error)]'
                      : isPinned
                        ? 'bg-primary/10'
                        : ''
                  }`}
                  title={
                    isMismatch
                      ? 'クリックで反例トレース(信号伝播+原因ゲート)を開始'
                      : 'クリックでこの入力パターンをピン留め'
                  }
                  onClick={() => (isMismatch ? startTrace(row.k) : pinPattern(row.k))}
                >
                  <td className="px-2 py-0.5 tracking-widest">
                    {row.inputBits.join('')}
                  </td>
                  <td className="px-2 py-0.5 tracking-widest">
                    {row.outputBits.join('')}
                  </td>
                  <td className="px-2 py-0.5">
                    <div className="flex gap-px">
                      {Array.from({ length: outWidth }, (_, j) => {
                        const bit = exp?.[j] ?? 0;
                        return (
                          <button
                            key={j}
                            type="button"
                            className={`flex h-[16px] w-[16px] cursor-pointer items-center justify-center rounded-[2px] border leading-none ${
                              bit
                                ? 'border-[#86efac] bg-[#dcfce7] font-bold text-[var(--status-ok)]'
                                : 'bg-background text-muted-foreground'
                            }`}
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpectedBit(row.k, j);
                            }}
                          >
                            {bit}
                          </button>
                        );
                      })}
                    </div>
                  </td>
                  <td className="px-2 py-0.5 text-center">
                    {info.perRow[i] ? (
                      <span style={{ color: 'var(--status-ok)' }}>○</span>
                    ) : (
                      <span style={{ color: 'var(--destructive)' }}>✖</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
