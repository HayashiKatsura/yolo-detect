import os
import random
import cv2

# 设置图片文件夹路径
image_folder = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_and_test_images/test_images/single_class/synthetic_200_normal"  # 替换为你的图片文件夹路径
output_folder = "/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_a_datasets/single_class/augment/single_anomaly_black_hole"  # 替换为输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 设定扣除区域的尺寸
cut_size = 40

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # 读取图片
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        continue  # 跳过无法读取的图片

    height, width, _ = image.shape

    # 生成随机扣除的矩形位置
    x_min = random.randint(0, width - cut_size)
    y_min = random.randint(0, height - cut_size)
    x_max = x_min + cut_size
    y_max = y_min + cut_size

    # 填充黑色
    image[y_min:y_max, x_min:x_max] = (0, 0, 0)

    # 计算YOLO格式标注信息（归一化）
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    w_norm = cut_size / width
    h_norm = cut_size / height

    # 生成文件名
    output_image_path = os.path.join(output_folder, image_file)
    label_file_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")

    # 保存修改后的图片
    cv2.imwrite(output_image_path, image)

    # 保存YOLO格式的标注文件
    with open(label_file_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("处理完成，所有图片和标注文件已保存到:", output_folder)
