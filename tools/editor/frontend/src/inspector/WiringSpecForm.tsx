// 構造化配線の引数フォーム。lut_layer の connections が rot / rot_n /
// butterfly / local1d / local2d のとき NodeParamForm から描画される。
// - 引数は文字列で編集し、適用時に ir/wiring.ts の変換で型付きの spec にする
// - 「配線を生成」で POST /api/wiring/generate(= core.connections.make_connections)を
//   呼び、返った indices を IR の lut_layer.params.indices へ書き込む(真理値表は保つ)。
//   あわせて train_info.connections / wiring を更新する(draft なら build 時に同じ関数で再生成)
// - 被覆率などの統計を 1 行で出す

import { useEffect, useState } from 'react';
import { ApiError, postWiringGenerate } from '../api/rest';
import {
  defaultWiringSpec,
  describeWiringStats,
  isStructuredConnection,
  specValueToText,
  textToSpecValue,
  WIRING_FIELDS,
  type WiringStats,
} from '../ir/wiring';
import type { LutLayerNodeData } from '../ir/irToGraph';
import { useGraphStore } from '../store/graphStore';

interface Props {
  nodeId: string;
  data: LutLayerNodeData;
}

export function WiringSpecForm({ nodeId, data }: Props) {
  const updateNodeData = useGraphStore((s) => s.updateNodeData);
  const updateLutParams = useGraphStore((s) => s.updateLutParams);
  const kind = data.connections;
  const fields = isStructuredConnection(kind) ? WIRING_FIELDS[kind] : [];

  const [texts, setTexts] = useState<Record<string, string>>({});
  const [dedupe, setDedupe] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<string | null>(null);

  // ノード / 種が変わったら入力欄を spec(無ければ既定値)から作り直す
  useEffect(() => {
    if (!isStructuredConnection(kind)) return;
    const spec = data.wiring ?? defaultWiringSpec(kind, data.nInputs);
    const next: Record<string, string> = {};
    for (const f of WIRING_FIELDS[kind]) next[f.key] = specValueToText(spec[f.key], f);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- ノード/種の切替に追従する意図的なリセット
    setTexts(next);
    setDedupe(Boolean(spec.dedupe));
    setError(null);
    setStats(null);
    // data.wiring は適用時にこのフォーム自身が更新するので依存に含めない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId, kind, data.nInputs]);

  if (!isStructuredConnection(kind)) return null;

  const parseSpec = (): Record<string, unknown> | null => {
    const spec: Record<string, unknown> = {};
    for (const f of fields) {
      const r = textToSpecValue(texts[f.key] ?? '', f);
      if (!r.ok) {
        setError(`${f.label} の形式が不正です(${f.hint})`);
        return null;
      }
      if (r.value !== undefined) spec[f.key] = r.value;
    }
    return spec;
  };

  const apply = async () => {
    const spec = parseSpec();
    if (!spec) return;
    setBusy(true);
    setError(null);
    try {
      const res = await postWiringGenerate({
        kind,
        in_width: data.inWidth,
        out_width: data.outWidth,
        n_inputs: data.nInputs,
        spec,
        dedupe,
      });
      // IR が真実源: indices を書き込み、種と引数を train_info に残す
      updateLutParams(nodeId, (p) => {
        p.indices = res.indices.map((row) => row.map(Number));
      });
      updateNodeData(nodeId, {
        connections: kind,
        wiring: { ...spec, ...(dedupe ? { dedupe: true } : {}) },
      });
      setStats(describeWiringStats(res.stats as WiringStats, data.inWidth));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wiring-form" data-testid="wiring-spec-form">
      <div className="form-row">
        {fields.map((f) => (
          <label key={f.key} title={f.hint}>
            {f.label}
            <input
              data-testid={`wiring-field-${f.key}`}
              value={texts[f.key] ?? ''}
              placeholder={f.optional ? '(省略可)' : ''}
              onChange={(e) => setTexts({ ...texts, [f.key]: e.target.value })}
            />
          </label>
        ))}
        <label title="同じ入力を複数スロットで読むゲートの重複を +1 ずらしで潰す">
          dedupe
          <input
            type="checkbox"
            data-testid="wiring-dedupe"
            checked={dedupe}
            onChange={(e) => setDedupe(e.target.checked)}
          />
        </label>
      </div>
      <div className="form-row">
        <button data-testid="wiring-generate" disabled={busy} onClick={() => void apply()}>
          {busy ? '生成中…' : '配線を生成して適用'}
        </button>
      </div>
      {stats && (
        <div className="hint-text" data-testid="wiring-stats">
          {stats}
        </div>
      )}
      {error && (
        <div className="warn-badge" data-testid="wiring-error">
          {error}
        </div>
      )}
      <div className="hint-text">
        引数の意味は core.connections.make_connections の docstring と同じ。
        生成後も真理値表は保たれ、draft ノードは build 時に同じ規則で再生成される。
      </div>
    </div>
  );
}
