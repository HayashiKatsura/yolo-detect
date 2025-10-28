# config.py

NORMAL_IMAGES_DIR = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_and_test_images/test_images/single_class/synthetic_200_normal"
ANOMALY_IMAGES_DIR = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_and_test_images/yolo_json/single_class"
ANOMALY_LABELS_DIR = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_and_test_images/yolo_json/single_class"

OUTPUT_IMAGES_DIR = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_a_datasets/single_class/augment/single_anomaly_all_ag_without_noise"
OUTPUT_LABELS_DIR = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_a_datasets/single_class/augment/single_anomaly_all_ag_without_noise"

NUM_SAMPLES = 200
MAX_RATIO = 0.15  # 异常区域最多占正常图比例
MIN_RATIO = 0.05  # 新增，异常区域最小宽高比例