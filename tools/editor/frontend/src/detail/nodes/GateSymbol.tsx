// 16ゲートのMIL/ANSI論理回路記号SVG。
// AND/OR/XOR/BUF系は distinctive-shape、比較系は入力側バブル、定数はカード表記。
// 塗り=ゲート色トークン、白背景での判別性はゲート名ラベル(GateNode側)で補完する。

import { rgbCss, type RGB } from '../../ir/gates';

type Shape = 'and' | 'or' | 'xor' | 'buf' | 'const';

interface SymbolSpec {
  shape: Shape;
  outBubble?: boolean;
  inBubbles?: [boolean, boolean]; // [slot0(A), slot1(B)]
  bufInput?: 0 | 1; // buf/notの参照スロット
  constLabel?: '0' | '1';
}

// ゲートID(N=2) → 記号仕様。A=slot0 / B=slot1
const SPECS: Record<number, SymbolSpec> = {
  0: { shape: 'const', constLabel: '0' }, // FALSE
  1: { shape: 'and' }, // AND
  2: { shape: 'and', inBubbles: [false, true] }, // A>B = A AND !B
  3: { shape: 'buf', bufInput: 0 }, // A
  4: { shape: 'and', inBubbles: [true, false] }, // B>A = !A AND B
  5: { shape: 'buf', bufInput: 1 }, // B
  6: { shape: 'xor' }, // XOR
  7: { shape: 'or' }, // OR
  8: { shape: 'or', outBubble: true }, // NOR
  9: { shape: 'xor', outBubble: true }, // XNOR
  10: { shape: 'buf', bufInput: 1, outBubble: true }, // !B
  11: { shape: 'or', inBubbles: [false, true] }, // A>=B = A OR !B
  12: { shape: 'buf', bufInput: 0, outBubble: true }, // !A
  13: { shape: 'or', inBubbles: [true, false] }, // B>=A = !A OR B
  14: { shape: 'and', outBubble: true }, // NAND
  15: { shape: 'const', constLabel: '1' }, // TRUE
};

const W = 52;
const H = 28;
const Y_A = 8; // slot0の入力線y
const Y_B = 20; // slot1の入力線y
const BODY_X = 12; // 記号本体の左端
const OUT_X = 40; // 記号本体の右端

interface Props {
  gateId: number;
  color: RGB;
}

/** N=2の16ゲート記号。N>2や未知IDは定数カードにフォールバック。 */
export function GateSymbol({ gateId, color }: Props) {
  const spec = SPECS[gateId & 15] ?? { shape: 'const' as const, constLabel: '0' as const };
  const fill = rgbCss(color, 0.85);
  const stroke = rgbCss(color);
  const line = 'var(--edge-default)';

  const body = (() => {
    switch (spec.shape) {
      case 'and':
        return (
          <path
            d={`M${BODY_X} 3 H${BODY_X + 14} A11 11 0 0 1 ${BODY_X + 14} 25 H${BODY_X} Z`}
            fill={fill}
            stroke={stroke}
          />
        );
      case 'or':
        return (
          <path
            d={`M${BODY_X - 2} 3 Q${BODY_X + 16} 3 ${OUT_X} 14 Q${BODY_X + 16} 25 ${BODY_X - 2} 25 Q${BODY_X + 5} 14 ${BODY_X - 2} 3 Z`}
            fill={fill}
            stroke={stroke}
          />
        );
      case 'xor':
        return (
          <>
            <path
              d={`M${BODY_X + 2} 3 Q${BODY_X + 18} 3 ${OUT_X} 14 Q${BODY_X + 18} 25 ${BODY_X + 2} 25 Q${BODY_X + 9} 14 ${BODY_X + 2} 3 Z`}
              fill={fill}
              stroke={stroke}
            />
            <path
              d={`M${BODY_X - 3} 3 Q${BODY_X + 4} 14 ${BODY_X - 3} 25`}
              fill="none"
              stroke={stroke}
            />
          </>
        );
      case 'buf':
        return (
          <path
            d={`M${BODY_X} 3 L${OUT_X - 4} 14 L${BODY_X} 25 Z`}
            fill={fill}
            stroke={stroke}
          />
        );
      case 'const':
        return (
          <>
            <rect x={BODY_X} y={4} width={22} height={20} rx={4} fill={fill} stroke={stroke} />
            <text
              x={BODY_X + 11}
              y={18}
              textAnchor="middle"
              fontSize={11}
              fontWeight={700}
              fill="#ffffff"
            >
              {spec.constLabel}
            </text>
          </>
        );
    }
  })();

  // 入力線: bufは参照スロットのみ、constは無し
  const inputLines =
    spec.shape === 'const' ? null : spec.shape === 'buf' ? (
      <>
        <line
          x1={0}
          y1={spec.bufInput === 1 ? Y_B : Y_A}
          x2={BODY_X}
          y2={spec.bufInput === 1 ? Y_B : Y_A}
          stroke={line}
        />
        <line
          x1={BODY_X}
          y1={spec.bufInput === 1 ? Y_B : Y_A}
          x2={BODY_X}
          y2={14}
          stroke={line}
        />
      </>
    ) : (
      <>
        <line x1={0} y1={Y_A} x2={BODY_X + 2} y2={Y_A} stroke={line} />
        <line x1={0} y1={Y_B} x2={BODY_X + 2} y2={Y_B} stroke={line} />
        {spec.inBubbles?.[0] && (
          <circle cx={BODY_X - 3} cy={Y_A} r={3} fill="var(--canvas-bg)" stroke={stroke} />
        )}
        {spec.inBubbles?.[1] && (
          <circle cx={BODY_X - 3} cy={Y_B} r={3} fill="var(--canvas-bg)" stroke={stroke} />
        )}
      </>
    );

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      {inputLines}
      {body}
      {spec.shape !== 'const' && (
        <line x1={OUT_X} y1={14} x2={W - (spec.outBubble ? 8 : 2)} y2={14} stroke={line} />
      )}
      {spec.outBubble && (
        <circle cx={W - 6} cy={14} r={3.2} fill="var(--canvas-bg)" stroke={stroke} />
      )}
    </svg>
  );
}
