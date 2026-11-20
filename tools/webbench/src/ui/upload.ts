// 手元の画像アップロードの共通処理(det系4画面+segで重複していた実装を集約)。
//
// 中央正方形切り出し(cover crop)は webcam_main.ts::frameRgba と同じ考え方
// (最大の中央正方形を切り出してから dst にリサイズ)。サンプル画像は元から正方形
// なので切り出しは恒等変換になるが、この集約はアップロード画像専用に閉じる
// (サンプル描画は drawSample 側の別経路のまま)。

/** 中央正方形の切り出し矩形(元画像の座標系)。 */
export interface CropRect {
  sx: number;
  sy: number;
  sSide: number;
}

/** 元画像の一辺 srcW×srcH から、中央の最大正方形を求める。 */
export function coverCropRect(srcW: number, srcH: number): CropRect {
  const sSide = Math.min(srcW, srcH);
  return { sx: (srcW - sSide) / 2, sy: (srcH - sSide) / 2, sSide };
}

/** CanvasImageSource の実寸(型ごとにプロパティ名が違うのでダックタイピングで拾う)。 */
function sourceSize(image: CanvasImageSource): { w: number; h: number } {
  if ('videoWidth' in image) return { w: image.videoWidth, h: image.videoHeight };
  if ('naturalWidth' in image) return { w: image.naturalWidth, h: image.naturalHeight };
  const sized = image as unknown as { width: number; height: number };
  return { w: sized.width, h: sized.height };
}

/** 任意の画像ソースを中央正方形に切り出して dst(dstW×dstH)へ描く。 */
export function drawCoverCrop(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  dstW: number,
  dstH: number,
): void {
  const { w, h } = sourceSize(image);
  const { sx, sy, sSide } = coverCropRect(w, h);
  ctx.drawImage(image, sx, sy, sSide, sSide, 0, 0, dstW, dstH);
}

/** RGBA(Uint8ClampedArray)をRec.601輝度でグレースケール化する(その場書き換え、
    r=g=bに揃える。alphaは変えない)。単チャンネルDLGNモデルにカラー写真を
    そのまま流すとR成分だけ見ることになる(detmono)ので、その回避に使う。 */
export function toGrayRgba(rgba: Uint8ClampedArray): void {
  for (let i = 0; i < rgba.length; i += 4) {
    const y = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    rgba[i] = y;
    rgba[i + 1] = y;
    rgba[i + 2] = y;
  }
}

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.avif', '.svg'];

/** file.type が空のドラッグ&ドロップ由来ファイルも拾えるよう、拡張子でも判定する。 */
export function isImageFile(file: { type: string; name: string }): boolean {
  if (file.type) return file.type.startsWith('image/');
  const lower = file.name.toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export interface WireImageUploadOptions {
  /** サイドバーの `<input type="file">`。 */
  fileInput: HTMLInputElement;
  /** 読み込みに成功したら呼ぶ。 */
  onImage: (img: HTMLImageElement) => void;
  /** 非画像ファイル・読み込み失敗のときに短い日本語メッセージで呼ぶ。 */
  onError: (message: string) => void;
  /** ドラッグ&ドロップを受け付ける対象(既定 document 全体)。 */
  dropTarget?: Document | HTMLElement;
}

/** ファイル入力のchange + ページ全体へのドラッグ&ドロップを1箇所にまとめる。
    object URL は読み込み成功・失敗いずれの後も revoke する(リークしない)。 */
export function wireImageUpload(opts: WireImageUploadOptions): void {
  const { fileInput, onImage, onError } = opts;
  const dropTarget = opts.dropTarget ?? document;

  function loadFile(f: File): void {
    if (!isImageFile(f)) {
      onError('画像として読み込めませんでした');
      return;
    }
    const img = new Image();
    const url = URL.createObjectURL(f);
    img.onload = () => {
      URL.revokeObjectURL(url);
      onImage(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      onError('画像として読み込めませんでした');
    };
    img.src = url;
  }

  fileInput.onchange = (ev) => {
    const f = (ev.target as HTMLInputElement).files?.[0];
    if (f) loadFile(f);
  };

  dropTarget.addEventListener('dragover', (ev) => {
    ev.preventDefault();
  });
  dropTarget.addEventListener('drop', (ev) => {
    ev.preventDefault();
    const f = (ev as DragEvent).dataTransfer?.files?.[0];
    if (f) loadFile(f);
  });
}
