// 真理値表セル編集。選択lut_layerノードのニューロン1個の
// 2^Nビットをグリッド表示し、セルクリックでトグルする。
// - 編集可否: train_info.parameterization × TT_EDITABLE(コア規約コピー)
// - ビルド済み(needsBuild=false)なら単一ニューロンAPIでin-place同期
//   (バックエンドがIRも更新するのでPUTは省略)。未ビルドはIR更新+デバウンスPUT
// - 訓練running中はバックエンドが409(pausedは許可)

import { useEffect, useState } from 'react';
import { postNodeTruthTable } from '../api/rest';
import { GATE_NAMES, ttToGateIdN2 } from '../ir/gates';
import { popcount, ttBit, ttParse, ttToggleBit } from '../ir/hexTT';
import type { LutLayerParams } from '../ir/schema';
import { useGraphStore } from '../store/graphStore';
import { useTrainingStore } from '../store/trainingStore';
import { useUIStore } from '../store/uiStore';

// コア規約のコピー(griffith_logic/ir/weights.py TT_EDITABLE)
const TT_EDITABLE = new Set(['softmax16g', 'independent', 'raw']);

// 編集不可パラメータ化の理由(ツールチップ表示)
const EDIT_BLOCK_REASONS: Record<string, string> = {
  iwp: '単一minterm構造のため編集不可',
  warp: 'WHT逆変換は将来UI対応',
  softmax: '表現力制約(softmax N>2は単一minterm/FALSEのみ表現可能)',
};

interface Props {
  nodeId: string;
}

/** グリッド列数: N=2→2、N=4→4、N=6→8(N=6は8×8)。 */
function gridCols(nInputs: number): number {
  return 1 << Math.ceil(nInputs / 2);
}

/** パターンkのスロット値表示: k=5, N=3 → "101"(s0が左)。 */
function patternBits(k: number, nInputs: number): string {
  let s = '';
  for (let j = 0; j < nInputs; j++) s += (k >> j) & 1;
  return s;
}

export function TruthTableEditor({ nodeId }: Props) {
  const ir = useGraphStore((s) => s.ir);
  const needsBuild = useGraphStore((s) => s.needsBuild);
  const updateLutParams = useGraphStore((s) => s.updateLutParams);
  const trainStatus = useTrainingStore((s) => s.status);

  const selectedNeuron = useUIStore((s) => s.selectedNeuron);

  const [neuron, setNeuron] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ノード切替時はニューロン選択とエラーをリセット
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- ノード切替時のローカル編集状態リセット(意図的なprop駆動setState)
    setNeuron(0);
    setError(null);
  }, [nodeId]);

  // 詳細ゲートビューでゲートをクリックしたら当該ニューロンへ連動
  useEffect(() => {
    if (selectedNeuron && selectedNeuron.nodeId === nodeId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- 詳細ビューのゲート選択(外部store)との連動(意図的)
      setNeuron(selectedNeuron.index);
      setError(null);
    }
  }, [selectedNeuron, nodeId]);

  const node = ir?.nodes.find((n) => n.id === nodeId);
  if (!node || node.type !== 'lut_layer') return null;

  const p = node.params as unknown as LutLayerParams;
  const nInputs = p.n_inputs;
  const nEntries = 1 << nInputs;
  const outWidth = p.out_width;
  const param = (node.train_info.parameterization as string) ?? 'softmax';
  const effective = param === 'softmax' && nInputs === 2 ? 'softmax16g' : param;
  const editable = TT_EDITABLE.has(effective);
  const blockReason = editable ? null : (EDIT_BLOCK_REASONS[param] ?? '編集不可');
  const running = trainStatus === 'running' || trainStatus === 'starting';

  const safeNeuron = Math.min(Math.max(neuron, 0), outWidth - 1);
  const hex = p.truth_tables[safeNeuron] ?? '0x0';
  const canEdit = editable && !busy && !running;

  const handleToggle = async (k: number) => {
    if (!canEdit) return;
    const next = ttToggleBit(hex, k);
    setError(null);
    if (!needsBuild) {
      // ビルド済み: バックエンドAPIがIR+weightsを同時更新(PUT不要)
      setBusy(true);
      try {
        await postNodeTruthTable(nodeId, safeNeuron, next);
        updateLutParams(
          nodeId,
          (params) => {
            params.truth_tables[safeNeuron] = next;
          },
          { sync: false },
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    } else {
      // 未ビルド(draft編集中): IRのみ更新し既存デバウンスPUTで同期
      updateLutParams(nodeId, (params) => {
        params.truth_tables[safeNeuron] = next;
      });
    }
  };

  const cols = gridCols(nInputs);
  const gateName =
    nInputs === 2 ? GATE_NAMES[ttToGateIdN2(Number(ttParse(hex) & 0xfn))] : null;

  return (
    <div className="tt-editor">
      <h3>真理値表編集</h3>
      <div className="form-row">
        <label>
          ニューロン番号 (0〜{outWidth - 1})
          <input
            type="number"
            min={0}
            max={outWidth - 1}
            value={safeNeuron}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (Number.isInteger(v)) setNeuron(v);
            }}
          />
        </label>
      </div>
      {!editable && (
        <div className="hint-text" title={blockReason ?? undefined}>
          🔒 {param}: {blockReason}
        </div>
      )}
      {running && (
        <div className="hint-text">訓練running中は編集不可(pauseで編集可能)</div>
      )}
      <div
        className={`tt-grid${canEdit ? '' : ' readonly'}`}
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
        title={blockReason ?? undefined}
      >
        {Array.from({ length: nEntries }, (_, k) => {
          const bit = ttBit(hex, k);
          return (
            <button
              key={k}
              type="button"
              className={`tt-cell${bit ? ' on' : ''}`}
              disabled={!canEdit}
              title={`k=${k} 入力(${patternBits(k, nInputs)}) → ${bit}`}
              onClick={() => void handleToggle(k)}
            >
              {bit}
            </button>
          );
        })}
      </div>
      <div className="hint-text">
        {hex} ・ popcount {popcount(hex)}/{nEntries}
        {gateName !== null && (
          <>
            {' '}
            ・ ゲート <b>{gateName}</b>
          </>
        )}
      </div>
      {error && <div className="error-text">{error}</div>}
    </div>
  );
}
