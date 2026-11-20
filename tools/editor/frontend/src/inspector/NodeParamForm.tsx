// 選択ノードのパラメータ編集フォーム(E2)。
// lut_layer: out_width/n_inputs/parameterization等(in_widthは接続から導出でreadonly、
//            構造変更はweights破棄=draft化)
// group_sum: k/tau、binarize: threshold、入力ポート: width

import { useState } from 'react';
import type {
  GenericNodeData,
  GroupSumNodeData,
  LutLayerNodeData,
} from '../ir/irToGraph';
import type { IRNodeJson, IRPortJson } from '../ir/schema';
import {
  CONNECTION_CHOICES,
  defaultWiringSpec,
  isStructuredConnection,
} from '../ir/wiring';
import { useGraphStore } from '../store/graphStore';
import { WiringSpecForm } from './WiringSpecForm';

const N_INPUTS_CHOICES = [2, 4, 6];
const PARAMETERIZATIONS = ['softmax', 'independent', 'iwp', 'raw', 'warp'];
const CONNECTIONS = CONNECTION_CHOICES;
const SELECTIONS = ['soft', 'hard_st'];

/** bias 欄の文字列(カンマ区切り整数)を長さ k の整数列へ。空は「バイアス無し」。不正は null */
function parseBias(text: string, k: number): number[] | undefined | null {
  const t = text.trim();
  if (t === '') return undefined;
  const nums = t.split(',').map((s) => Number(s.trim()));
  if (nums.length !== k || nums.some((v) => !Number.isInteger(v))) return null;
  return nums;
}

// train_info.kind の表示ラベルと、DWN選択時の学習method提案。
const KIND_LABEL: Record<string, string> = {
  dwn: 'DWN',
  binarybrain: 'BinaryBrain',
  learnable_mapping: 'LearnableMapping',
  embedding: 'Embedding',
};

/** train_info.kind バッジ+method提案。kind未設定なら何も描画しない。*/
function KindBadge({ node }: { node: IRNodeJson | undefined }) {
  const kind = node?.train_info?.kind;
  if (typeof kind !== 'string' || !kind) return null;
  const label = KIND_LABEL[kind] ?? kind;
  return (
    <div className="flex flex-col gap-1">
      <span
        className="draft-badge"
        data-testid="node-kind-badge"
        title="train_info.kind(抽出器由来の部品種別)"
      >
        種別: {label}
      </span>
      {kind === 'dwn' && (
        <div className="hint-text" data-testid="dwn-method-hint">
          学習method提案: DWN系(ste)を推奨。学習設定のmethodセレクタで選択してください。
        </div>
      )}
    </div>
  );
}

/** group_sum の判定バイアス欄。長さ k の整数列、空なら無し。HDL / C は未対応 */
function GroupSumBiasField({
  k,
  bias,
  onChange,
}: {
  k: number;
  bias: number[] | undefined;
  onChange: (bias: number[] | undefined) => void;
}) {
  const [text, setText] = useState(bias ? bias.join(',') : '');
  const parsed = parseBias(text, k);
  return (
    <div className="form-row">
      <label className="grow" title="クラスごとの整数の判定バイアス(τ で割る前の整数和に足す)">
        bias(整数×{k}、空=無し)
        <input
          data-testid="group-sum-bias"
          value={text}
          placeholder="例: 0,1"
          onChange={(e) => {
            const next = e.target.value;
            setText(next);
            const b = parseBias(next, k);
            if (b !== null) onChange(b);
          }}
        />
      </label>
      {parsed === null && (
        <div className="warn-badge" data-testid="group-sum-bias-error">
          長さ {k} の整数列(カンマ区切り)が必要です
        </div>
      )}
      {parsed && parsed.some((v) => v !== 0) && (
        <div className="hint-text">bias 付きの IR は HDL / C 生成では拒否されます</div>
      )}
    </div>
  );
}

function parsePositiveInt(value: string): number | null {
  const n = Number(value);
  return Number.isInteger(n) && n >= 1 ? n : null;
}

function parseBits(value: string, width: number): number[] | null {
  const trimmed = value.trim();
  if (trimmed.length !== width || /[^01]/.test(trimmed)) return null;
  return Array.from(trimmed, (ch) => Number(ch));
}

export function NodeParamForm() {
  const selectedId = useGraphStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const ir = useGraphStore((s) => s.ir);
  const updateNodeData = useGraphStore((s) => s.updateNodeData);
  const updatePortWidth = useGraphStore((s) => s.updatePortWidth);
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams);
  const deleteSelected = useGraphStore((s) => s.deleteSelected);

  if (!ir || !selectedId) return null;

  const irNode = ir.nodes.find((n) => n.id === selectedId);

  const inputPort: IRPortJson | undefined = ir.inputs.find(
    (p) => p.id === selectedId,
  );
  if (inputPort) {
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            width(入力ビット幅)
            <input
              type="number"
              min={1}
              value={inputPort.width}
              onChange={(e) => {
                const w = parsePositiveInt(e.target.value);
                if (w !== null) updatePortWidth(selectedId, w);
              }}
            />
          </label>
        </div>
        <div className="hint-text">幅変更で下流lut_layerのweightsは破棄されます</div>
      </div>
    );
  }

  const flowNode = nodes.find((n) => n.id === selectedId);
  if (!flowNode) return null;

  if (flowNode.type === 'lutLayer') {
    const d = flowNode.data as LutLayerNodeData;
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            in_width(導出)
            <input value={d.inWidth} readOnly disabled />
          </label>
          <label>
            out_width
            <input
              type="number"
              min={1}
              value={d.outWidth}
              onChange={(e) => {
                const w = parsePositiveInt(e.target.value);
                if (w !== null) updateNodeData(selectedId, { outWidth: w });
              }}
            />
          </label>
          <label>
            n_inputs
            <select
              value={d.nInputs}
              onChange={(e) =>
                updateNodeData(selectedId, { nInputs: Number(e.target.value) })
              }
            >
              {N_INPUTS_CHOICES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="form-row">
          <label>
            parameterization
            <select
              value={d.parameterization}
              onChange={(e) =>
                updateNodeData(selectedId, { parameterization: e.target.value })
              }
            >
              {PARAMETERIZATIONS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label>
            connections
            <select
              data-testid="lut-connections"
              value={d.connections}
              onChange={(e) => {
                const kind = e.target.value;
                // 構造化配線は引数の既定値を一緒に持たせ、random / unique は引数を捨てる
                updateNodeData(selectedId, {
                  connections: kind,
                  wiring: isStructuredConnection(kind)
                    ? defaultWiringSpec(kind, d.nInputs)
                    : undefined,
                });
              }}
            >
              {CONNECTIONS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label>
            selection
            <select
              value={d.selection}
              onChange={(e) =>
                updateNodeData(selectedId, { selection: e.target.value })
              }
            >
              {SELECTIONS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>
        {isStructuredConnection(d.connections) && (
          <WiringSpecForm key={`${selectedId}:${d.connections}`} nodeId={selectedId} data={d} />
        )}
        {d.initialized ? (
          <div className="warn-badge">
            ⚠ out_width / n_inputs / parameterization の変更でweightsは破棄されます
          </div>
        ) : (
          <div className="draft-badge">draft(未学習・buildで確定)</div>
        )}
        <KindBadge node={irNode} />
        <button className="danger" onClick={deleteSelected}>
          ノードを削除
        </button>
      </div>
    );
  }

  if (flowNode.type === 'groupSum') {
    const d = flowNode.data as GroupSumNodeData;
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            in_width(導出)
            <input value={d.inWidth} readOnly disabled />
          </label>
          <label>
            k(クラス数)
            <input
              type="number"
              min={1}
              value={d.k}
              onChange={(e) => {
                const k = parsePositiveInt(e.target.value);
                if (k !== null) updateNodeData(selectedId, { k });
              }}
            />
          </label>
          <label>
            tau
            <input
              type="number"
              step="0.1"
              value={d.tau}
              onChange={(e) => {
                const tau = Number(e.target.value);
                if (Number.isFinite(tau) && tau > 0) {
                  updateNodeData(selectedId, { tau });
                }
              }}
            />
          </label>
        </div>
        <GroupSumBiasField
          key={`${selectedId}:${d.k}`}
          k={d.k}
          bias={d.bias}
          onChange={(bias) => updateNodeData(selectedId, { bias })}
        />
        <button className="danger" onClick={deleteSelected}>
          ノードを削除
        </button>
      </div>
    );
  }

  const d = flowNode.data as GenericNodeData;
  if (d.kind === 'blackbox' && irNode) {
    return (
      <BlackboxParamForm
        key={selectedId}
        node={irNode}
        onChange={(patch) => updateNodeParams(selectedId, patch)}
        onDelete={deleteSelected}
      />
    );
  }
  if (d.kind === 'binarize') {
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            width(導出)
            <input value={d.width ?? '?'} readOnly disabled />
          </label>
          <label>
            threshold
            <input
              type="number"
              step="0.1"
              value={d.threshold ?? 0.5}
              onChange={(e) => {
                const t = Number(e.target.value);
                if (Number.isFinite(t)) {
                  updateNodeData(selectedId, { threshold: t });
                }
              }}
            />
          </label>
        </div>
        <button className="danger" onClick={deleteSelected}>
          ノードを削除
        </button>
      </div>
    );
  }

  if (d.kind === 'flatten') {
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            width(導出)
            <input value={d.width ?? '?'} readOnly disabled />
          </label>
        </div>
        <div className="hint-text">上流テンソルを平坦なビット/特徴ベクトルとして扱います</div>
        <button className="danger" onClick={deleteSelected}>
          ノードを削除
        </button>
      </div>
    );
  }

  if (d.kind === 'constant') {
    const width = d.width ?? 1;
    const bits = Array.isArray(d.bits)
      ? (d.bits as number[]).map((b) => (b ? 1 : 0))
      : new Array<number>(width).fill(0);
    return (
      <div className="param-form">
        <div className="form-row">
          <label>
            width
            <input
              type="number"
              min={1}
              value={width}
              onChange={(e) => {
                const w = parsePositiveInt(e.target.value);
                if (w !== null) {
                  updateNodeData(selectedId, {
                    width: w,
                    bits: new Array<number>(w).fill(0),
                  });
                }
              }}
            />
          </label>
          <label>
            bits
            <input
              value={bits.join('')}
              onChange={(e) => {
                const next = parseBits(e.target.value, width);
                if (next !== null) updateNodeData(selectedId, { bits: next });
              }}
            />
          </label>
        </div>
        <div className="hint-text">入力を持たない固定ビット列です</div>
        <button className="danger" onClick={deleteSelected}>
          ノードを削除
        </button>
      </div>
    );
  }

  return null;
}

// blackboxノードの汎用パラメータエディタ。
// torch_class(文字列)と config(JSON)を編集し、contract(in/out幅)は表示のみ。
// key={nodeId} で選択切替時にローカル状態(JSONテキスト)をリセットする。
function BlackboxParamForm({
  node,
  onChange,
  onDelete,
}: {
  node: IRNodeJson;
  onChange: (patch: Record<string, unknown>) => void;
  onDelete: () => void;
}) {
  const p = node.params as Record<string, unknown>;
  const torchClass = typeof p.torch_class === 'string' ? p.torch_class : '';
  const inWidth = typeof p.in_width === 'number' ? p.in_width : '?';
  const outWidth = typeof p.out_width === 'number' ? p.out_width : '?';
  const contract = (node.train_info?.contract ?? null) as Record<string, string> | null;

  const [configText, setConfigText] = useState(() =>
    JSON.stringify(p.config ?? {}, null, 2),
  );
  const [configError, setConfigError] = useState<string | null>(null);

  const onConfigChange = (text: string) => {
    setConfigText(text);
    try {
      const parsed = JSON.parse(text);
      if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setConfigError('configはJSONオブジェクトが必要です');
        return;
      }
      setConfigError(null);
      onChange({ config: parsed });
    } catch {
      setConfigError('JSONの構文エラー');
    }
  };

  return (
    <div className="param-form">
      <div className="form-row">
        <label>
          torch_class
          <input
            data-testid="blackbox-torch-class"
            value={torchClass}
            onChange={(e) => onChange({ torch_class: e.target.value })}
          />
        </label>
      </div>
      <div className="form-row">
        <label>
          in_width(contract・表示のみ)
          <input value={inWidth} readOnly disabled />
        </label>
        <label>
          out_width(contract・表示のみ)
          <input value={outWidth} readOnly disabled />
        </label>
      </div>
      <div className="form-row">
        <label style={{ width: '100%' }}>
          config(JSON)
          <textarea
            data-testid="blackbox-config"
            rows={6}
            style={{ width: '100%', fontFamily: 'monospace', fontSize: '12px' }}
            value={configText}
            onChange={(e) => onConfigChange(e.target.value)}
          />
        </label>
      </div>
      {configError && (
        <div className="warn-badge" data-testid="blackbox-config-error">
          ⚠ {configError}
        </div>
      )}
      {contract && (
        <div className="hint-text" data-testid="blackbox-contract">
          contract: {contract.input} → {contract.output}
          {contract.hdl ? ` / ${contract.hdl}` : ''}
        </div>
      )}
      <button className="danger" onClick={onDelete}>
        ノードを削除
      </button>
    </div>
  );
}
