import os
import sys

from matplotlib import colors
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

import os
import cv2
import random


def generate_random_colors(n):
    """生成n个深色随机颜色"""
    colors = []
    for i in range(n):
        # 限制RGB值范围在0-0.7之间，确保颜色较深
        r = random.uniform(0, 0.7)
        g = random.uniform(0, 0.7)
        b = random.uniform(0, 0.7)
        # 确保至少有一个通道的值较高，以保持颜色可见度
        max_channel = random.randint(0, 2)
        if max_channel == 0:
            r = random.uniform(0.5, 0.9)
        elif max_channel == 1:
            g = random.uniform(0.5, 0.9)
        else:
            b = random.uniform(0.5, 0.9)
        colors.append((r, g, b))
    return colors

# 设置文件夹路径
image_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/images/val'
label_folder = [{
                'name': 'true_labels',
                'label':"/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/labels/val",
                'save':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/compare'},
                {'name': 'v12-damage',
                'label': '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/predict/predict_results_conf0.25',
                'save':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/compare'}
                ]


# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
# colors = generate_random_colors(len(label_folder))
colors = [
    #绿色 true_labels
    (0, 1, 0),
    #红色 v8
    (1, 0, 0), 
]
# 处理每张图片
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    for index,item in enumerate(label_folder):
        label_path = os.path.join(item['label'], os.path.splitext(image_file)[0] + ".txt")
        output_path = os.path.join(item['save'], image_file)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_file}")
            continue

        height, width, _ = image.shape

        # 读取标注文件
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                count = 0
                putText = False
                for line in file.readlines():
                    count = len(file.readlines())
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w, h = map(float, parts)

                        # 计算边界框的坐标
                        x1 = int((x_center - w / 2) * width)
                        y1 = int((y_center - h / 2) * height)
                        x2 = int((x_center + w / 2) * width)
                        y2 = int((y_center + h / 2) * height)
                        
                        
                        # 画矩形框
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        if not putText:
                            cv2.putText(image, str(count), (5, 5*index+10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 2)
                            putText = True
                        # cv2.putText(image, str(class_id), (x1, y1 - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 2)

        # 保存标注后的图片
        cv2.imwrite(output_path, image)

