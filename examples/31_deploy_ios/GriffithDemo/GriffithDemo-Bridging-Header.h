/* Swift から C を直接呼ぶための bridging header。
 * モデル固有のヘッダ(model.h)はここに入れない: モデルごとに typedef が衝突するため、
 * 共通の記述子 gd_model(GdModel.h)だけを見せる。 */
#include "GdModel.h"
