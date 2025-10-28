import os
import random

import sys 
sys.path.append("/home/panxiang/Documents/kweilxfilebox/ultralytics")

from _project.mydata._code._augment.config import *
from _project.mydata._code._augment.utils.labelme_utils import get_largest_region_and_crop
from _project.mydata._code._augment.utils.augment_utils import augment_anomaly, augment_final_image, save_bbox_stats,record_bbox_stats
from _project.mydata._code._augment.utils.paste_utils import resize_anomaly_to_fit, paste_anomaly
from _project.mydata._code._augment.utils.io_utils import save_image_and_label
from PIL import Image

def main():
    normal_imgs = [os.path.join(NORMAL_IMAGES_DIR, f) for f in os.listdir(NORMAL_IMAGES_DIR) if str(f).lower().endswith('.jpg') or str(f).lower().endswith('.png')]
    anomaly_imgs = [os.path.join(ANOMALY_IMAGES_DIR, f) for f in os.listdir(ANOMALY_IMAGES_DIR) if str(f).lower().endswith('.jpg') or str(f).lower().endswith('.png')]
    anomaly_jsons = [os.path.join(ANOMALY_LABELS_DIR, f) for f in os.listdir(ANOMALY_LABELS_DIR) if str(f).lower().endswith('.json')]
    normal_imgs = sorted(normal_imgs)
    anomaly_imgs = sorted(anomaly_imgs)
    anomaly_jsons = sorted(anomaly_jsons)
    
    for i in range(NUM_SAMPLES):
        # normal_img = random.choice(normal_imgs)
        normal_img = normal_imgs[i]
        idx = random.randint(0, len(anomaly_imgs)-1)
        anomaly_img, anomaly_json = anomaly_imgs[idx], anomaly_jsons[idx]

        # 抠最大异常区域
        cropped_anomaly = get_largest_region_and_crop(anomaly_img, anomaly_json)
        # cropped_anomaly.save('cropped_anomaly.png')
        if cropped_anomaly is None:
            print("Skip: No valid anomaly region found.")
            continue

        # resize
        normal_w, normal_h = Image.open(normal_img).size
        # cropped_anomaly = resize_anomaly_to_fit(normal_w, normal_h, cropped_anomaly, MAX_RATIO)
        # resize
        cropped_anomaly = resize_anomaly_to_fit(
            normal_w=normal_w,
            normal_h=normal_h,
            anomaly_img=cropped_anomaly,
            max_ratio=MAX_RATIO,  # 保留已有
            min_ratio=MIN_RATIO   # 新增的，来自 config.py
        )
        if cropped_anomaly is None:
            print("Skip: Anomaly region too small after resize.")
            continue
        
        # 第一步，异常区域增强
        aug_anomaly_np, aug_bbox = augment_anomaly(cropped_anomaly)
        if aug_anomaly_np is None or aug_bbox is None:
            print("Skip: Anomaly region too small after augmentation.")
            continue

        # 第二步，贴图 + 计算最终 bbox
        final_img, yolo_label = paste_anomaly(normal_img, aug_anomaly_np, aug_bbox)

        # 第三步，整体图像增强（无 bbox）
        # final_img = augment_final_image(final_img)
        record_bbox_stats(aug_bbox, final_img.width, final_img.height)

        # 保存
        os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
        os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
        output_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"gen_{i}.jpg")
        output_label_path = os.path.join(OUTPUT_LABELS_DIR, f"gen_{i}.txt")
        save_image_and_label(final_img, yolo_label, output_img_path, output_label_path)


        print(f"[{i+1}/{NUM_SAMPLES}] Saved: {output_img_path}")
    
    # ==== 最后统计图保存 ====
    save_bbox_stats()  # 【新增】

if __name__ == "__main__":
    main()
