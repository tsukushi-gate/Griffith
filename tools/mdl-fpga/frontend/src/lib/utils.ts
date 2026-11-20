// 出典: tools/editor/frontend/src/lib/utils.ts(コピー)
// shadcn/ui標準のクラス結合ヘルパ

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
