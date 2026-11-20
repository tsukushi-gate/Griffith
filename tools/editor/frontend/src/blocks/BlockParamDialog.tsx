// 複合ブロックのパラメータダイアログ。
// パレットからのドロップ(blockDialogStore.pending)で開き、確定時に
// generateBlock → graphStore.addSubgraph でドロップ位置へサブグラフを展開する。

import { useState } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { createDraftIdAllocator, useGraphStore } from '../store/graphStore';
import { useBlockDialogStore } from './blockDialogStore';
import { INIT_PRESETS, INIT_PRESET_LABEL, type InitPreset } from './nodeInit';
import {
  ATTENTION_MODES,
  BLOCK_DEFS,
  CONV_DEFS,
  POSITIONAL_CODES,
  PARTS_DEFS,
  TRANSFORMER_MERGES,
  TRANSFORMER_NORMS,
  generateBlock,
  positionalCodeValid,
  type BlockParams,
  type PositionalCode,
} from './templates';
import { RECURRENT_CELL_TYPES, type RecurrentCellType } from './recurrentUnit';

/** "16,8" 形式の幅列をパースする(空欄は空配列=隠れ層なし) */
function parseWidthList(text: string): number[] | null {
  const t = text.trim();
  if (t === '') return [];
  const widths = t.split(/[,、\s]+/).map(Number);
  if (widths.some((w) => !Number.isInteger(w) || w < 1)) return null;
  return widths;
}

function parsePositiveInt(text: string): number | null {
  const n = Number(text);
  return Number.isInteger(n) && n >= 1 ? n : null;
}

function parseNonNegInt(text: string): number | null {
  const n = Number(text);
  return Number.isInteger(n) && n >= 0 ? n : null;
}

export function BlockParamDialog() {
  const pending = useBlockDialogStore((s) => s.pending);
  const close = useBlockDialogStore((s) => s.close);
  const addSubgraph = useGraphStore((s) => s.addSubgraph);

  // 種別ごとのフォーム状態(文字列で保持し確定時にパース)
  const [inWidth, setInWidth] = useState('8');
  const [hidden, setHidden] = useState('16');
  const [outWidth, setOutWidth] = useState('8');
  const [width, setWidth] = useState('8');
  const [innerLayers, setInnerLayers] = useState('1');
  const [merge, setMerge] = useState<'or' | 'xor'>('xor');
  const [k, setK] = useState('10');
  const [tau, setTau] = useState('1.0');
  // 初期化プリセット。lut系/convに反映される。
  const [preset, setPreset] = useState<InitPreset>('random');
  // conv2d
  const [inChannels, setInChannels] = useState('1');
  const [inH, setInH] = useState('8');
  const [inW, setInW] = useState('8');
  const [outChannels, setOutChannels] = useState('4');
  const [kernel, setKernel] = useState('3');
  const [stride, setStride] = useState('1');
  const [padding, setPadding] = useState('0');
  const [treeDepth, setTreeDepth] = useState('2');
  const [nInputs, setNInputs] = useState('2');
  // or_pool2d
  const [poolKernel, setPoolKernel] = useState('2');
  const [poolStride, setPoolStride] = useState('2');
  const [poolPadding, setPoolPadding] = useState('0');
  // reduction_tree
  const [rtInWidth, setRtInWidth] = useState('64');
  const [rtFanin, setRtFanin] = useState('6');
  const [rtMinWidth, setRtMinWidth] = useState('6');
  // モデル部品
  const [dModel, setDModel] = useState('8');
  const [seqLen, setSeqLen] = useState('4');
  const [nHeads, setNHeads] = useState('1');
  const [attnMode, setAttnMode] = useState<(typeof ATTENTION_MODES)[number]>('soft');
  const [tfNorm, setTfNorm] = useState<(typeof TRANSFORMER_NORMS)[number]>('layernorm');
  const [tfMerge, setTfMerge] = useState<(typeof TRANSFORMER_MERGES)[number]>('xor');
  const [peInWidth, setPeInWidth] = useState('8');
  const [peSeqLen, setPeSeqLen] = useState('4');
  const [peCodeWidth, setPeCodeWidth] = useState('3');
  const [peCode, setPeCode] = useState<PositionalCode>('binary');
  const [embVocab, setEmbVocab] = useState('16');
  const [embDim, setEmbDim] = useState('8');
  const [dwnIn, setDwnIn] = useState('16');
  const [dwnOut, setDwnOut] = useState('16');
  const [dwnNInputs, setDwnNInputs] = useState('6');
  const [dwnMapping, setDwnMapping] = useState<'random' | 'learnable'>('random');
  const [usC, setUsC] = useState('1');
  const [usH, setUsH] = useState('4');
  const [usW, setUsW] = useState('4');
  const [usScale, setUsScale] = useState('2');
  // recurrent(LSTM/GRU unroll)
  const [recCell, setRecCell] = useState<RecurrentCellType>('lstm');
  const [recUnroll, setRecUnroll] = useState('3');
  const [recInWidth, setRecInWidth] = useState('4');
  const [recHidden, setRecHidden] = useState('4');
  const [recDepth, setRecDepth] = useState('1');

  // フォーム値は意図的にリセットしない(直前値を保持して連続配置しやすくする)
  if (!pending) return null;
  const def =
    BLOCK_DEFS.find((d) => d.kind === pending.kind) ??
    CONV_DEFS.find((d) => d.kind === pending.kind) ??
    PARTS_DEFS.find((d) => d.kind === pending.kind);
  if (!def) return null;

  // 初期化プリセットが効くのは lut系ブロックと conv2d(or_pool2dは結線のみ)。
  const presetApplies =
    pending.kind === 'mlp' ||
    pending.kind === 'residual' ||
    pending.kind === 'classifier' ||
    pending.kind === 'conv2d' ||
    pending.kind === 'reduction_tree' ||
    pending.kind === 'embedding';

  // Attention/Transformerは blackbox(エディタのIR-build学習経路が不可)。
  const isBlackboxPart =
    pending.kind === 'attention' || pending.kind === 'transformer';

  const buildSpec = (): BlockParams | null => {
    if (pending.kind === 'mlp') {
      const i = parsePositiveInt(inWidth);
      const h = parseWidthList(hidden);
      const o = parsePositiveInt(outWidth);
      if (i === null || h === null || o === null) return null;
      return { kind: 'mlp', params: { inWidth: i, hidden: h, outWidth: o } };
    }
    if (pending.kind === 'residual') {
      const w = parsePositiveInt(width);
      const n = parsePositiveInt(innerLayers);
      if (w === null || n === null) return null;
      return { kind: 'residual', params: { width: w, innerLayers: n, merge } };
    }
    if (pending.kind === 'conv2d') {
      const ic = parsePositiveInt(inChannels);
      const h = parsePositiveInt(inH);
      const w = parsePositiveInt(inW);
      const oc = parsePositiveInt(outChannels);
      const kr = parsePositiveInt(kernel);
      const st = parsePositiveInt(stride);
      const pd = parseNonNegInt(padding);
      const td = parsePositiveInt(treeDepth);
      const ni = parsePositiveInt(nInputs);
      if ([ic, h, w, oc, kr, st, pd, td, ni].some((v) => v === null)) return null;
      // 2^n>=4(hex真理値表)の制約
      if ((ni as number) < 2) return null;
      return {
        kind: 'conv2d',
        params: {
          inChannels: ic!,
          inH: h!,
          inW: w!,
          outChannels: oc!,
          kernel: kr!,
          stride: st!,
          padding: pd!,
          treeDepth: td!,
          nInputs: ni!,
        },
      };
    }
    if (pending.kind === 'or_pool2d') {
      const ic = parsePositiveInt(inChannels);
      const h = parsePositiveInt(inH);
      const w = parsePositiveInt(inW);
      const kr = parsePositiveInt(poolKernel);
      const st = parsePositiveInt(poolStride);
      const pd = parseNonNegInt(poolPadding);
      if ([ic, h, w, kr, st, pd].some((v) => v === null)) return null;
      return {
        kind: 'or_pool2d',
        params: { inChannels: ic!, inH: h!, inW: w!, kernel: kr!, stride: st!, padding: pd! },
      };
    }
    if (pending.kind === 'reduction_tree') {
      const iw = parsePositiveInt(rtInWidth);
      const fn = parsePositiveInt(rtFanin);
      const mw = parsePositiveInt(rtMinWidth);
      if (iw === null || fn === null || mw === null || fn < 2) return null;
      return { kind: 'reduction_tree', params: { inWidth: iw, fanin: fn, minWidth: mw } };
    }
    if (pending.kind === 'attention') {
      const dm = parsePositiveInt(dModel);
      const sl = parsePositiveInt(seqLen);
      const nh = parsePositiveInt(nHeads);
      if (dm === null || sl === null || nh === null) return null;
      if (dm % nh !== 0) return null; // 射影次元がヘッド数で割り切れること
      return { kind: 'attention', params: { dModel: dm, seqLen: sl, nHeads: nh, mode: attnMode } };
    }
    if (pending.kind === 'transformer') {
      const dm = parsePositiveInt(dModel);
      const sl = parsePositiveInt(seqLen);
      const nh = parsePositiveInt(nHeads);
      if (dm === null || sl === null || nh === null) return null;
      if (dm % nh !== 0) return null;
      return {
        kind: 'transformer',
        params: { dModel: dm, seqLen: sl, nHeads: nh, norm: tfNorm, merge: tfMerge },
      };
    }
    if (pending.kind === 'positional_encoding') {
      const iw = parsePositiveInt(peInWidth);
      const sl = parsePositiveInt(peSeqLen);
      const cw = parsePositiveInt(peCodeWidth);
      if (iw === null || sl === null || cw === null) return null;
      if (!positionalCodeValid(sl, cw, peCode)) return null; // コード制約
      return {
        kind: 'positional_encoding',
        params: { inWidth: iw, seqLen: sl, codeWidth: cw, code: peCode },
      };
    }
    if (pending.kind === 'embedding') {
      const v = parsePositiveInt(embVocab);
      const d = parsePositiveInt(embDim);
      if (v === null || d === null) return null;
      return { kind: 'embedding', params: { vocab: v, dim: d } };
    }
    if (pending.kind === 'dwn') {
      const iw = parsePositiveInt(dwnIn);
      const ow = parsePositiveInt(dwnOut);
      const ni = parsePositiveInt(dwnNInputs);
      if (iw === null || ow === null || ni === null || ni < 2) return null;
      return {
        kind: 'dwn',
        params: { inWidth: iw, outWidth: ow, nInputs: ni, mapping: dwnMapping },
      };
    }
    if (pending.kind === 'upsample2d') {
      const c = parsePositiveInt(usC);
      const h = parsePositiveInt(usH);
      const w = parsePositiveInt(usW);
      const s = parsePositiveInt(usScale);
      if (c === null || h === null || w === null || s === null || s < 2) return null;
      return { kind: 'upsample2d', params: { inC: c, inH: h, inW: w, scale: s } };
    }
    if (pending.kind === 'recurrent') {
      const t = parsePositiveInt(recUnroll);
      const iw = parsePositiveInt(recInWidth);
      const hd = parsePositiveInt(recHidden);
      const dp = parsePositiveInt(recDepth);
      if (t === null || iw === null || hd === null || dp === null) return null;
      return {
        kind: 'recurrent',
        params: { cellType: recCell, unroll: t, inWidth: iw, hidden: hd, depth: dp },
      };
    }
    const i = parsePositiveInt(inWidth);
    const kk = parsePositiveInt(k);
    const t = Number(tau);
    if (i === null || kk === null || !Number.isFinite(t) || t <= 0) return null;
    return { kind: 'classifier', params: { inWidth: i, k: kk, tau: t } };
  };

  const submit = () => {
    const spec = buildSpec();
    if (!spec) {
      toast.error('パラメータが不正です(正整数/幅列/n_inputs≧2を確認してください)');
      return;
    }
    const tpl = generateBlock(spec, createDraftIdAllocator(), preset);
    addSubgraph(tpl.nodes, tpl.edges, pending.position);
    close();
    toast.success(
      `${def.label} を配置しました(入口: ${tpl.entryId} / 出口: ${tpl.exitId})`,
    );
  };

  return (
    <Dialog open onOpenChange={(open) => !open && close()}>
      <DialogContent className="max-w-sm">
        <DialogHeader>
          <DialogTitle>{def.label} のパラメータ</DialogTitle>
          <DialogDescription>{def.desc}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {pending.kind === 'mlp' && (
            <>
              <FieldRow label="入力幅" value={inWidth} onChange={setInWidth} />
              <div className="flex flex-col gap-1">
                <Label className="text-xs">隠れ層の幅列(カンマ区切り、空=なし)</Label>
                <Input
                  value={hidden}
                  onChange={(e) => setHidden(e.target.value)}
                  placeholder="例: 16,8"
                />
              </div>
              <FieldRow label="出力幅" value={outWidth} onChange={setOutWidth} />
            </>
          )}
          {pending.kind === 'residual' && (
            <>
              <FieldRow label="幅(全slot同幅)" value={width} onChange={setWidth} />
              <FieldRow label="内側の層数" value={innerLayers} onChange={setInnerLayers} />
              <div className="flex flex-col gap-1">
                <Label className="text-xs">マージ演算</Label>
                <select
                  className="h-8 rounded-md border bg-transparent px-2 text-sm"
                  value={merge}
                  onChange={(e) => setMerge(e.target.value as 'or' | 'xor')}
                >
                  <option value="xor">xor</option>
                  <option value="or">or</option>
                </select>
              </div>
            </>
          )}
          {pending.kind === 'classifier' && (
            <>
              <FieldRow label="入力幅" value={inWidth} onChange={setInWidth} />
              <FieldRow label="クラス数 k" value={k} onChange={setK} />
              <FieldRow label="tau" value={tau} onChange={setTau} type="text" />
            </>
          )}
          {pending.kind === 'conv2d' && (
            <>
              <div className="grid grid-cols-3 gap-2">
                <FieldRow label="入力C" value={inChannels} onChange={setInChannels} />
                <FieldRow label="入力H" value={inH} onChange={setInH} />
                <FieldRow label="入力W" value={inW} onChange={setInW} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <FieldRow label="出力チャネル" value={outChannels} onChange={setOutChannels} />
                <FieldRow label="カーネル" value={kernel} onChange={setKernel} />
                <FieldRow label="ストライド" value={stride} onChange={setStride} />
                <FieldRow label="パディング" value={padding} onChange={setPadding} />
                <FieldRow label="木の深さ" value={treeDepth} onChange={setTreeDepth} />
                <FieldRow label="LUT入力数 n" value={nInputs} onChange={setNInputs} />
              </div>
            </>
          )}
          {pending.kind === 'or_pool2d' && (
            <>
              <div className="grid grid-cols-3 gap-2">
                <FieldRow label="入力C" value={inChannels} onChange={setInChannels} />
                <FieldRow label="入力H" value={inH} onChange={setInH} />
                <FieldRow label="入力W" value={inW} onChange={setInW} />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <FieldRow label="カーネル" value={poolKernel} onChange={setPoolKernel} />
                <FieldRow label="ストライド" value={poolStride} onChange={setPoolStride} />
                <FieldRow label="パディング" value={poolPadding} onChange={setPoolPadding} />
              </div>
              <div className="hint-text">窓k×k は≦6入力(k≦2)まで。</div>
            </>
          )}
          {pending.kind === 'reduction_tree' && (
            <>
              <FieldRow label="入力幅" value={rtInWidth} onChange={setRtInWidth} />
              <div className="grid grid-cols-2 gap-2">
                <FieldRow label="fanin(縮約率)" value={rtFanin} onChange={setRtFanin} />
                <FieldRow label="最小幅(停止)" value={rtMinWidth} onChange={setRtMinWidth} />
              </div>
            </>
          )}
          {(pending.kind === 'attention' || pending.kind === 'transformer') && (
            <>
              <div className="grid grid-cols-3 gap-2">
                <FieldRow label="d_model" value={dModel} onChange={setDModel} />
                <FieldRow label="系列長 S" value={seqLen} onChange={setSeqLen} />
                <FieldRow label="ヘッド数" value={nHeads} onChange={setNHeads} />
              </div>
              {pending.kind === 'attention' && (
                <div className="flex flex-col gap-1">
                  <Label className="text-xs">mode</Label>
                  <select
                    data-testid="attn-mode-select"
                    className="h-8 rounded-md border bg-transparent px-2 text-sm"
                    value={attnMode}
                    onChange={(e) =>
                      setAttnMode(e.target.value as (typeof ATTENTION_MODES)[number])
                    }
                  >
                    {ATTENTION_MODES.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {pending.kind === 'transformer' && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs">norm</Label>
                    <select
                      className="h-8 rounded-md border bg-transparent px-2 text-sm"
                      value={tfNorm}
                      onChange={(e) =>
                        setTfNorm(e.target.value as (typeof TRANSFORMER_NORMS)[number])
                      }
                    >
                      {TRANSFORMER_NORMS.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1">
                    <Label className="text-xs">merge</Label>
                    <select
                      className="h-8 rounded-md border bg-transparent px-2 text-sm"
                      value={tfMerge}
                      onChange={(e) =>
                        setTfMerge(e.target.value as (typeof TRANSFORMER_MERGES)[number])
                      }
                    >
                      {TRANSFORMER_MERGES.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </>
          )}
          {pending.kind === 'positional_encoding' && (
            <>
              <FieldRow label="入力幅(上流)" value={peInWidth} onChange={setPeInWidth} />
              <div className="grid grid-cols-2 gap-2">
                <FieldRow label="系列長 S" value={peSeqLen} onChange={setPeSeqLen} />
                <FieldRow label="コード幅" value={peCodeWidth} onChange={setPeCodeWidth} />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs">コード種別</Label>
                <select
                  data-testid="pe-code-select"
                  className="h-8 rounded-md border bg-transparent px-2 text-sm"
                  value={peCode}
                  onChange={(e) => setPeCode(e.target.value as PositionalCode)}
                >
                  {POSITIONAL_CODES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
              <div className="hint-text">
                binary/gray: S≦2^コード幅 / thermometer: コード幅≧S-1。定数線は入力へ連結されます。
              </div>
            </>
          )}
          {pending.kind === 'embedding' && (
            <>
              <div className="grid grid-cols-2 gap-2">
                <FieldRow label="語彙数(vocab)" value={embVocab} onChange={setEmbVocab} />
                <FieldRow label="埋め込み次元" value={embDim} onChange={setEmbDim} />
              </div>
              <div className="hint-text">
                簡易実装: one-hot入力(幅=vocab)前提の lut_layer(vocab→dim)。
              </div>
            </>
          )}
          {pending.kind === 'dwn' && (
            <>
              <div className="grid grid-cols-3 gap-2">
                <FieldRow label="入力幅" value={dwnIn} onChange={setDwnIn} />
                <FieldRow label="出力幅(LUT数)" value={dwnOut} onChange={setDwnOut} />
                <FieldRow label="LUT入力数 n" value={dwnNInputs} onChange={setDwnNInputs} />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs">mapping</Label>
                <select
                  className="h-8 rounded-md border bg-transparent px-2 text-sm"
                  value={dwnMapping}
                  onChange={(e) => setDwnMapping(e.target.value as 'random' | 'learnable')}
                >
                  <option value="random">random</option>
                  <option value="learnable">learnable</option>
                </select>
              </div>
              <div className="hint-text">
                train_info.kind=dwn を付与します。学習は DWN系(ste)の method を推奨。
              </div>
            </>
          )}
          {pending.kind === 'upsample2d' && (
            <>
              <div className="grid grid-cols-4 gap-2">
                <FieldRow label="C" value={usC} onChange={setUsC} />
                <FieldRow label="H" value={usH} onChange={setUsH} />
                <FieldRow label="W" value={usW} onChange={setUsW} />
                <FieldRow label="倍率" value={usScale} onChange={setUsScale} />
              </div>
              <div className="hint-text">nearest拡大=恒等配線(n_inputs=1, tt=0x2)。</div>
            </>
          )}
          {pending.kind === 'recurrent' && (
            <>
              <div className="flex flex-col gap-1">
                <Label className="text-xs">セル種別</Label>
                <select
                  data-testid="recurrent-cell-select"
                  className="h-8 rounded-md border bg-transparent px-2 text-sm"
                  value={recCell}
                  onChange={(e) => setRecCell(e.target.value as RecurrentCellType)}
                >
                  {RECURRENT_CELL_TYPES.map((c) => (
                    <option key={c} value={c}>
                      {c.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <FieldRow label="unroll数 T" value={recUnroll} onChange={setRecUnroll} />
                <FieldRow label="ゲート段数 depth" value={recDepth} onChange={setRecDepth} />
                <FieldRow label="入力幅/step" value={recInWidth} onChange={setRecInWidth} />
                <FieldRow label="隠れ状態幅" value={recHidden} onChange={setRecHidden} />
              </div>
              <div className="hint-text">
                セルをT回時間展開(重み非共有)。入力幅=T×入力幅/step、出力=最終隠れ状態(幅=隠れ状態幅)。
                hard評価で scan_cell とビット一致(gates=draft・build時学習)。
              </div>
            </>
          )}
          {isBlackboxPart && (
            <div className="warn-badge" data-testid="blackbox-train-note">
              ⚠ blackboxノードです。エディタ内のIRビルド学習経路では学習できません
              (配置・形状診断・統計は可・HDL生成は不可)。学習は Python import 経路
              (models/*.py)のモデルで行ってください。
            </div>
          )}
          {presetApplies && (
            <div className="flex flex-col gap-1">
              <Label className="text-xs">初期化(Initializer)</Label>
              <select
                data-testid="init-preset-select"
                className="h-8 rounded-md border bg-transparent px-2 text-sm"
                value={preset}
                onChange={(e) => setPreset(e.target.value as InitPreset)}
              >
                {INIT_PRESETS.map((p) => (
                  <option key={p} value={p}>
                    {INIT_PRESET_LABEL[p]}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={close}>
            キャンセル
          </Button>
          <Button onClick={submit}>配置</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function FieldRow({
  label,
  value,
  onChange,
  type = 'number',
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Label className="text-xs">{label}</Label>
      <Input type={type} min={1} value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
