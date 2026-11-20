// editor frontendからの再export点(WB-4)。コピーせず相対importする
// (editor側のリファクタで壊れた場合はequivalence.test.tsが即検知する)。
// gates.ts はテーマ依存があるためimportしない。
export { evalCircuit } from '../../editor/frontend/src/playground/evalCircuit';
export { ttParse, ttBit } from '../../editor/frontend/src/ir/hexTT';
export { edgeSlot } from '../../editor/frontend/src/ir/schema';
export type {
  AddParams,
  ConcatParams,
  Conv2dParams,
  GroupSumParams,
  IREdgeJson,
  IRGraphJson,
  IRNodeJson,
  LutLayerParams,
  OrPool2dParams,
  PermuteParams,
  UpSample2dParams,
} from '../../editor/frontend/src/ir/schema';
