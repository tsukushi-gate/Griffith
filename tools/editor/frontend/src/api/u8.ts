// u8量子化エンベロープのデコーダ(WS/REST共通、backend monitor.py encode_u8 の逆変換)

import type { U8Envelope } from './rest';

// 同一エンベロープの重複デコードを回避するメモ。
// 詳細ビューでは同一層の全GateNodeが同じdistエンベロープをデコードするため、
// 1フレームにつき1回に抑える。返り値は共有インスタンスなので呼び出し側は
// 読み取り専用で扱うこと(既存呼び出し元は全て非破壊を確認済み)。
const decodeMemo = new WeakMap<U8Envelope, Float32Array>();

/** base64のu8データをFloat32Arrayへ復元する(値 = q/255 × scale)。 */
export function decodeU8(env: U8Envelope): Float32Array {
  const hit = decodeMemo.get(env);
  if (hit !== undefined) return hit;
  const raw = atob(env.data);
  const out = new Float32Array(raw.length);
  const k = env.scale / 255;
  for (let i = 0; i < raw.length; i++) {
    out[i] = raw.charCodeAt(i) * k;
  }
  decodeMemo.set(env, out);
  return out;
}
