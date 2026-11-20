// 未実装トップタブのプレースホルダ。EVALUATION/DATASET/CONFIGは
// 後続系列(N5/N3/N4)で本実装に差し替える。クラッシュせず「実装予定」を表示する。
// バッジ文言はi18n辞書を参照して言語切替に追従する。

import { useT } from '../i18n';

interface PlaceholderTabProps {
  title: string;
  description: string;
}

export function PlaceholderTab({ title, description }: PlaceholderTabProps) {
  const t = useT();
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <h2 className="text-lg font-semibold text-muted-foreground">{title}</h2>
      <p className="max-w-md text-sm text-muted-foreground/70">{description}</p>
      <span className="rounded-full border px-3 py-0.5 text-xs text-muted-foreground/60">
        {t('placeholder.plannedBadge')}
      </span>
    </div>
  );
}
