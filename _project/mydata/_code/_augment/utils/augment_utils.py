import albumentations as A
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt  # 新增

# 统计数据容器
bbox_widths = []
bbox_heights = []
bbox_areas = []
bbox_centers_x = []
bbox_centers_y = []


# ----------- 统计函数 -----------
def record_bbox_stats(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    center_x = (x_min + x_max) / 2 / img_w  # 归一化
    center_y = (y_min + y_max) / 2 / img_h  # 归一化

    bbox_widths.append(width)
    bbox_heights.append(height)
    bbox_areas.append(area)
    bbox_centers_x.append(center_x)
    bbox_centers_y.append(center_y)


def save_bbox_stats(output_dir='./stats'):
    os.makedirs(output_dir, exist_ok=True)

    # 宽度分布
    plt.hist(bbox_widths, bins=30)
    plt.title("BBox Width Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bbox_width_distribution.png'))
    plt.clf()

    # 高度分布
    plt.hist(bbox_heights, bins=30)
    plt.title("BBox Height Distribution")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bbox_height_distribution.png'))
    plt.clf()

    # 面积分布
    plt.hist(bbox_areas, bins=30)
    plt.title("BBox Area Distribution")
    plt.xlabel("Area (pixels^2)")
    plt.ylabel("Count")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bbox_area_distribution.png'))
    plt.clf()

    # 中心位置
    plt.scatter(bbox_centers_x, bbox_centers_y, alpha=0.5)
    plt.title("BBox Center Distribution (Normalized)")
    plt.xlabel("X Center")
    plt.ylabel("Y Center")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'bbox_center_distribution.png'))
    plt.clf()




def augment_anomaly(anomaly_img):
    if anomaly_img.size[0] < 1 or anomaly_img.size[1] < 1:
        print("Warning: Anomaly region too small for augmentation, skipping.")
        return None, None

    w, h = anomaly_img.size  # PIL 格式 (w, h)

    transform = A.Compose([
        # 几何类变换
        A.Rotate(limit=20, p=0.5),  # 随机旋转（最大±20度），让异常区域有不同角度
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),  # 随机平移、缩放、旋转，模拟不同位置和大小的异常
        A.HorizontalFlip(p=0.5),  # 水平翻转，生成左右对称异常
        A.VerticalFlip(p=0.5),  # 垂直翻转，生成上下对称异常
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # 弹性扭曲，模拟柔性材料或液体异常形态
        A.GridDistortion(p=0.3),  # 网格扭曲，模拟变形效果
        A.Perspective(scale=(0.05, 0.1), p=0.3),  # 透视变化，模拟不同拍摄角度的异常区域

        # 颜色类变换
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # 随机调整亮度、对比度、饱和度、色调
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # 调整色相、饱和度、亮度
        # A.Solarize(threshold=128, p=0.2),  # 高于阈值的像素取反，产生特殊效果，模拟特殊异常
        # A.RandomBrightnessContrast(p=0.5),  # 随机亮度和对比度调整
        # A.RandomGamma(gamma_limit=(80, 120), p=0.5),  # Gamma变化，调整整体亮暗风格

        # 噪声类变换
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 高斯噪声，模拟图像采集噪声
        # A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),  # 乘法噪声，随机放大或缩小像素值，模拟局部干扰
        # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # 相机ISO噪声，模拟弱光场景的噪点

    ], 
    bbox_params=A.BboxParams(
        format='pascal_voc',  # Pascal VOC 格式，(x_min, y_min, x_max, y_max)
        label_fields=['category_ids'],  # 配合标签一起变换
        min_visibility=0.3  # 至少保留 30% 的目标区域，防止异常区域被完全裁掉
    ))
    
    
    transformed = transform(
        image=np.array(anomaly_img),
        bboxes=[[0, 0, w, h]],
        category_ids=[0]
    )

    # 增强后 bbox 检查
    if len(transformed['bboxes']) == 0:
        print("Warning: Bbox disappeared after augmentation, skipping.")
        return None, None

    # 安全取 bbox
    x_min, y_min, x_max, y_max = transformed['bboxes'][0]

    # clip 到有效范围
    x_min = max(0, min(x_min, w))
    x_max = max(0, min(x_max, w))
    y_min = max(0, min(y_min, h))
    y_max = max(0, min(y_max, h))

    # 无效 bbox (面积为0)，跳过
    if x_max - x_min < 1 or y_max - y_min < 1:
        print("Warning: Augmented bbox invalid (zero area), skipping.")
        return None, None

    return transformed['image'], [x_min, y_min, x_max, y_max]

def augment_final_image(full_image):
    transform_image = A.Compose([
        # A.CoarseDropout(max_holes=5, max_height=16, max_width=16, p=0.3),  # 无 bbox -> 会产生黑块
        # A.Cutout(num_holes=3, max_h_size=10, max_w_size=10, fill_value=0, p=0.3), -> 会产生黑块
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 可重复使用也行
    ])

    result = transform_image(image=np.array(full_image))
    return Image.fromarray(result['image'])
