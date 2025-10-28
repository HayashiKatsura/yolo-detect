from PIL import Image
import numpy as np
import random

# def resize_anomaly_to_fit(normal_w, normal_h, anomaly_img, max_ratio=0.5):
#     aw, ah = anomaly_img.size
#     max_w, max_h = int(normal_w * max_ratio), int(normal_h * max_ratio)
#     if aw > max_w or ah > max_h:
#         scale = min(max_w / aw, max_h / ah)
#         new_w, new_h = int(aw * scale), int(ah * scale)
#         # 兼容 Pillow 新旧版本
#         try:
#             resample = Image.Resampling.LANCZOS
#         except AttributeError:
#             resample = Image.ANTIALIAS
#         anomaly_img = anomaly_img.resize((new_w, new_h), resample)
#     return anomaly_img

def resize_anomaly_to_fit(normal_w, normal_h, anomaly_img, max_ratio=0.5, min_ratio=0.1):
    aw, ah = anomaly_img.size  # PIL 的宽高

    # -------- 防止除0异常 ----------
    if aw == 0 or ah == 0:
        print("Warning: Anomaly region size is zero, skipping resize.")
        return None

    # 最大最小宽高限制
    max_w, max_h = int(normal_w * max_ratio), int(normal_h * max_ratio)
    min_w, min_h = int(normal_w * min_ratio), int(normal_h * min_ratio)

    # 计算缩放比例
    scale = 1.0
    if aw > max_w or ah > max_h:
        scale = min(max_w / aw, max_h / ah)  # 缩小比例
    elif aw < min_w or ah < min_h:
        scale = max(min_w / aw, min_h / ah)  # 放大比例

    # 新尺寸
    new_w, new_h = int(aw * scale), int(ah * scale)

    # 保险处理，防止算完后变成 0
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 调整尺寸
    resized_anomaly = anomaly_img.resize((new_w, new_h), Image.Resampling.LANCZOS)  # 注意 PIL >= 10 用 Resampling
    return resized_anomaly



def paste_anomaly(normal_img_path, anomaly_np, anomaly_bbox):
    normal_img = Image.open(normal_img_path).convert("RGB")
    nw, nh = normal_img.size
    aw, ah = anomaly_np.shape[1], anomaly_np.shape[0]

    # 随机粘贴位置
    x_offset, y_offset = np.random.randint(0, nw - aw), np.random.randint(0, nh - ah)

    # 转 PIL
    anomaly_pil = Image.fromarray(anomaly_np)

    # 直接贴图
    normal_img.paste(anomaly_pil, (x_offset, y_offset))

    # bbox 坐标调整
    x_min, y_min, x_max, y_max = anomaly_bbox
    x_min += x_offset
    y_min += y_offset
    x_max += x_offset
    y_max += y_offset

    # 转 YOLO 格式
    x_center = ((x_min + x_max) / 2) / nw
    y_center = ((y_min + y_max) / 2) / nh
    width = (x_max - x_min) / nw
    height = (y_max - y_min) / nh
    yolo_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    return normal_img, yolo_label
