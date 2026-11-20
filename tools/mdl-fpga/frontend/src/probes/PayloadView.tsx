// プローブ payload の描画(バックエンドの payload 契約と 1:1)。
// bits はビット順(既定 [msb:lsb])と横幅(cols)を args で制御できる。
// queue は過去値のスタック(新しい順、未充填は x プレースホルダ)。
import type { ProbeArgs, ProbePayload } from '../api/types';
import { hexToBits } from '../graph/hexBits';

const BIT_CELL_PX = 12; // .bit-cell 10px + gap 2px

export function bitCells(
  hex: string,
  width: number,
  order: 'msb' | 'lsb',
): { index: number; on: boolean }[] {
  const bits = hexToBits(hex, width);
  const idx = Array.from({ length: width }, (_, i) => i);
  if (order === 'msb') idx.reverse();
  return idx.map((i) => ({ index: i, on: !!bits[i] }));
}

function BitsView({ hex, width, args }: {
  hex: string; width: number; args: ProbeArgs;
}) {
  const order = args.order === 'lsb' ? 'lsb' : 'msb'; // 既定は慣例 [msb:lsb]
  const cols = args.cols && args.cols > 0
    ? Math.min(args.cols, width) : undefined;
  return (
    <div
      className="payload-bits"
      style={cols ? { width: cols * BIT_CELL_PX, maxWidth: 'none' } : undefined}
      title={order === 'msb' ? '[msb:lsb]' : '[lsb:msb]'}
    >
      {bitCells(hex, width, order).map((c) => (
        <div
          key={c.index}
          className={`bit-cell${c.on ? ' on' : ''}`}
          title={`bit ${c.index} = ${c.on ? 1 : 0}`}
        />
      ))}
    </div>
  );
}

export function PayloadView({ payload, args }: {
  payload: ProbePayload | undefined;
  args?: ProbeArgs;
}) {
  const a = args ?? {};
  if (!payload) return <div className="payload-empty">値なし(step してください)</div>;
  switch (payload.kind) {
    case 'error':
      return <div className="payload-error">⚠ {payload.message}</div>;
    case 'text':
      return <div className="payload-text">{payload.text}</div>;
    case 'bits':
      return <BitsView hex={payload.hex} width={payload.width} args={a} />;
    case 'bars': {
      const max = Math.max(1, ...payload.values);
      return (
        <div>
          <div className="payload-bars">
            {payload.values.map((v, i) => (
              <div key={i} className="barwrap">
                <div
                  className={`bar${i === payload.argmax ? ' top' : ''}`}
                  style={{ height: `${Math.max(3, (v / max) * 72)}px` }}
                  title={`${payload.labels?.[i] ?? i}: ${v}`}
                />
                <span>{payload.labels?.[i] ?? i}</span>
              </div>
            ))}
          </div>
          {payload.argmax !== null && (
            <div className="payload-argmax">
              argmax = {payload.labels?.[payload.argmax] ?? payload.argmax}
              ({payload.values[payload.argmax]})
            </div>
          )}
        </div>
      );
    }
    case 'image':
      return (
        <img
          className="payload-image"
          src={payload.data_url}
          width={Math.min(224, payload.w * 8)}
          alt={`${payload.w}x${payload.h}`}
        />
      );
    case 'queue': {
      // 新しい順に depth 段。未充填スロットは x
      const slots = Array.from({ length: payload.depth }, (_, i) =>
        payload.items[i] ?? null);
      return (
        <div className="payload-queue">
          {slots.map((item, i) => (
            <div key={i} className="queue-slot">
              <span className="queue-cycle">
                {item ? `c${item.cycle}` : 'x'}
              </span>
              {item ? (
                <PayloadView payload={item.payload} args={a} />
              ) : (
                <span className="queue-empty">x</span>
              )}
            </div>
          ))}
        </div>
      );
    }
    default:
      return <div className="payload-error">未知の payload</div>;
  }
}
