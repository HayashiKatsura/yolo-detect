1. 颜色类操作（不影响 bbox 的位置）
操作名	说明
Equalize	直方图均衡化
Solarize	像素值超过阈值取反
SolarizeAdd	对小于128的像素增加值
Contrast	调整对比度
Color	调整颜色饱和度
Brightness	调整亮度
Sharpness	调整锐度
2. 几何类操作（同时调整 bbox）
改变整张图片的位置、形状、角度等，从而让模型学会在各种视角、偏移下识别目标。
同时必须调整对应的 BBox，否则目标的位置不对
操作	意义	举例（形象解释）
Rotate	旋转图像，比如旋转 15°	把猫图顺时针转 15°，bbox 也要跟着转
TranslateX	水平平移，整体往左或右移动	把整个图往右移动 20 像素，bbox 的 x 坐标也要+20
TranslateY	垂直平移，整体往上或下移动	把图像整体往下移动，bbox 跟着动
ShearX	水平剪切，把图像变成平行四边形形状	图像左右拉扯，bbox 也变形
ShearY	垂直剪切，上下拉伸成平行四边形	图像上下拉扯，bbox 也要跟着变
Cutout	随机 遮挡一部分区域	用灰色块随机挡住图像的一部分，增强模型鲁棒性（bbox 通常不调整，但目标可能被遮挡）
作名	说明
ShearX	水平剪切
ShearY	垂直剪切
TranslateX	水平平移
TranslateY	垂直平移
Rotate	旋转
Cutout	随机区域遮挡（灰色）
3. BBox Only 操作（只影响 bbox 内像素，不改 bbox 本身）
只改变 bbox 框内的内容，不影响背景和整体图像。
类似给目标做个“小手术”，让目标外观变化，但不动位置。
操作	意义	举例（形象解释）
BBox Only Equalize	只对 bbox 内做 直方图均衡化，增强对比度	让框内的猫颜色对比度变高，外面不变
BBox Only Rotate	只旋转框内内容，不动 bbox	猫的头转一圈，背景不动
BBox Only FlipLR	左右翻转框内内容，不动 bbox	框内猫左右翻个面，位置不动
BBox Only TranslateX/Y	只在框内 平移目标，像“移动里面的东西”	框定了猫，猫在框内挪动，框位置不动
操作名	说明
BBox Only Equalize	仅 bbox 区域做均衡化
BBox Only Rotate	仅 bbox 区域旋转
BBox Only FlipLR	仅 bbox 区域左右翻转
BBox Only TranslateX/Y	仅 bbox 区域平移


异常图 + labelme -> 抠最大区域 -> resize fit 正常图 -> 增强 Albumentations (返回增强图 + bbox)
                      ↓
                  (异常区域增强后图)
                      ↓
         贴到正常图 + 随机位置偏移 (x, y)
                      ↓
          新 bbox = 增强 bbox + (x, y)
                      ↓
     转换为 YOLO 格式 (归一化中心点 + 宽高)
                      ↓
         保存图像 + txt 标签
