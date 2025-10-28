def generate_csv(images_list, labels_dict, output_dir, csv_name):
    """生成比较标签的CSV表格"""
    csv_path = os.path.join(output_dir, csv_name)
    
    # 创建CSV表头
    headers = ["file_name"]
    
    # 添加真实标签的表头 (xt, yt, wt, ht)
    headers.extend(["xt", "yt", "wt", "ht"])
    
    # 添加预测标签的表头 (x0, y0, w0, h0, x1, y1, w1, h1, ...)
    for i in range(1, len(labels_dict)):
        headers.extend([f"x{i-1}", f"y{i-1}", f"w{i-1}", f"h{i-1}"])
    
    # 打开CSV文件准备写入
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # 处理两组测试图像
        for img_idx, imgs_classes in enumerate(images_list):
            data_class = img_idx  # 0 或 1
            
            # 处理异常图像文件夹
            anomaly_dir = os.path.join(imgs_classes, '_anomaly')
            if os.path.exists(anomaly_dir):
                process_images_for_csv(anomaly_dir, data_class, labels_dict, writer)
            
            # 处理普通图像文件夹中的图像
            process_images_for_csv(imgs_classes, data_class, labels_dict, writer)
    
    print(f"CSV文件已生成: {csv_path}")
    return csv_path

def process_images_for_csv(img_dir, data_class, labels_dict, writer):
    """处理文件夹中的图像，提取标签并写入CSV"""
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')) and not os.path.isdir(os.path.join(img_dir, img_file)):
            img_name = os.path.splitext(img_file)[0]
            
            # 为CSV文件添加前缀以区分数据类别
            file_name = f"data{data_class}_{img_name}"
            
            # 收集所有标签源的标签
            all_labels = []
            max_labels_count = 0
            
            for label_source in labels_dict:
                label_path = label_source[f'data{data_class}']
                boxes = parse_labels(label_path, img_name)
                all_labels.append(boxes)
                max_labels_count = max(max_labels_count, len(boxes))
            
            # 如果没有标签，则跳过该图像
            if max_labels_count == 0:
                continue
            
            # 填充标签数据到CSV
            for label_idx in range(max_labels_count):
                row = [file_name]
                
                # 添加每个标签源的数据
                for source_idx, boxes in enumerate(all_labels):
                    if label_idx < len(boxes):
                        box = boxes[label_idx]
                        # 真实标签通常有5个值: 类别 x y w h
                        # 预测标签可能有6个值: 类别 x y w h 置信度
                        if source_idx == 0:  # 真实标签
                            if len(box) >= 5:
                                row.extend(box[1:5])  # 添加 x y w h
                            else:
                                row.extend(["", "", "", ""])
                        else:  # 预测标签
                            if len(box) >= 5:
                                row.extend(box[1:5])  # 添加 x y w h
                            else:
                                row.extend(["", "", "", ""])
                    else:
                        # 如果该标签源没有足够的标签，添加空值
                        row.extend(["", "", "", ""])
                
                # 写入一行数据
                writer.writerow(row)
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='标签可视化与CSV生成工具')
    parser.add_argument('--output_dir', type=str, default="/home/panxiang/coding/kweilx/ultralytics/_project/label_and_test_images/visualization_results",
                        help='输出目录路径')
    parser.add_argument('--csv_name', type=str, default="labels_comparison.csv",
                        help='输出CSV文件名')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='置信度阈值')
    return parser.parse_args()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import csv
import argparse

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

def draw_box(img, box, color, thickness=2):
    """在图像上绘制边界框"""
    # 假设box格式为 [类别, x, y, w, h]，其中坐标为归一化坐标
    x, y, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
    
    # 将归一化坐标转换为像素坐标
    height, width, _ = img.shape
    x1 = int((x - w/2) * width)
    y1 = int((y - h/2) * height)
    x2 = int((x + w/2) * width)
    y2 = int((y + h/2) * height)
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def parse_labels(label_path, img_name):
    """解析标签文件"""
    boxes = []
    label_file = os.path.join(label_path, f"{img_name}.txt")
    
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                boxes.append(parts)
    return boxes

def main(labels_list=None, base_output_dir=None, csv_path=None):
    """主函数，支持直接传参或通过命令行参数获取配置"""
    # 如果没有直接传参，则通过命令行获取
    # if labels_list is None or base_output_dir is None:
    #     args = parse_args()
    #     base_output_dir = args.output_dir
    #     csv_name = args.csv_name
    #     conf_threshold = args.conf_threshold
    # else:
    #     # 如果直接传参，使用传入的值
    #     conf_threshold = 0.25
    #     csv_name = os.path.basename(csv_path) if csv_path else "labels_comparison.csv"
    
    # 确保labels_list是列表类型
    if labels_list and isinstance(labels_list, tuple):
        labels_list = list(labels_list)
    
    # 定义图像路径
    images_list = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/label_and_test_images/test_images/single_class/medical0611/0",
        "/home/panxiang/coding/kweilx/ultralytics/_project/label_and_test_images/test_images/single_class/medical0611/1"
    ]
    
    # 定义保存结果的路径，为两组数据分别创建文件夹
    # base_output_dir = args.output_dir
    output_dirs = [
        os.path.join(base_output_dir, "data0_results"),
        os.path.join(base_output_dir, "data1_results")
    ]
    
    # 创建输出目录
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 主输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 定义真实标签路径
    labels_dict = [{
        'name': 'TrueLabels',
        'data0': "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/medical/202506111525/labels/train",
        'data1': "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/medical/202506111525/labels/val"
    }]
    
    # 如果没有传入labels_list，使用默认值
    # if not labels_list:
    #     # 默认的预测标签路径列表
    #     labels_list = [
    #         "/home/panxiang/coding/kweilx/ultralytics/_project/predict_results/algorithm1",
    #         "/home/panxiang/coding/kweilx/ultralytics/_project/predict_results/algorithm2",
    #     ]
    
    # 添加预测标签，使用命令行传入的置信度阈值
    # conf_threshold = args.conf_threshold
    
    # 预测标签
    conf_threshold = 0.25
    for item in labels_list:
        labels_dict.append({
            'name': os.path.basename(item),
            'data0': os.path.join(item, f'DATA_0_CONF_{conf_threshold}/predict_labels'),
            'data1': os.path.join(item, f'DATA_1_CONF_{conf_threshold}/predict_labels')
        })
    
    # 生成随机颜色
    # colors = generate_random_colors(len(labels_dict))
    colors = [
        
        # 红色
        (1, 0, 0),
        
        # 绿色
        (0, 1, 0),
        
    ]
    
    # 生成CSV文件
    if csv_path:
        csv_file = csv_path
    else:
        csv_file = os.path.join(base_output_dir, 'labels_comparison.csv')
    
    csv_path = generate_csv(images_list, labels_dict, base_output_dir, os.path.basename(csv_file))
    print(f"生成CSV文件: {csv_path}")
    
    # 处理两组测试图像
    for img_idx, imgs_classes in enumerate(images_list):
        # 确定数据类别
        data_class = img_idx  # 0 或 1
        
        # 处理异常图像文件夹
        anomaly_dir = os.path.join(imgs_classes, '_anomaly')
        if os.path.exists(anomaly_dir):
            for img_file in os.listdir(anomaly_dir):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_name = os.path.splitext(img_file)[0]
                    img_path = os.path.join(anomaly_dir, img_file)
                    
                    # 读取图像
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
                    
                    # 创建绘图
                    fig, ax = plt.subplots(1, figsize=(12, 8))
                    ax.imshow(img)
                    
                    # 添加图例条目
                    legend_handles = []
                    
                    # 处理每组标签
                    for label_idx, label_source in enumerate(labels_dict):
                        label_path = label_source[f'data{data_class}']
                        
                        # 解析标签文件
                        boxes = parse_labels(label_path, img_name)
                        
                        color = colors[label_idx]
                        color_rgb = tuple(int(c * 255) for c in color)
                        
                        # 绘制边界框
                        for box in boxes:
                            if len(box) >= 5:  # 确保box格式正确
                                # 添加到图像上
                                x, y, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                                
                                # 转换归一化坐标到像素坐标
                                height, width, _ = img.shape
                                x_center = x * width
                                y_center = y * height
                                box_width = w * width
                                box_height = h * height
                                
                                # 创建矩形
                                rect = patches.Rectangle(
                                    (x_center - box_width/2, y_center - box_height/2),
                                    box_width, box_height,
                                    linewidth=2,
                                    edgecolor=color,
                                    facecolor='none'
                                )
                                ax.add_patch(rect)
                        
                        # 添加图例
                        legend_handles.append(patches.Patch(color=color, label=label_source['name']))
                    
                    # 显示图例
                    ax.legend(handles=legend_handles, loc='upper left')
                    
                    # 关闭坐标轴
                    ax.axis('off')
                    
                    # 保存图像到对应数据集的文件夹
                    save_path = os.path.join(output_dirs[data_class], f"{img_name}.png")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    
                    print(f"Saved visualization to {save_path}")
        
        # 处理普通图像文件夹
        for img_file in os.listdir(imgs_classes):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')) and not os.path.isdir(os.path.join(imgs_classes, img_file)):
                img_name = os.path.splitext(img_file)[0]
                img_path = os.path.join(imgs_classes, img_file)
                
                # 读取图像
                img = cv2.imread(img_path)
                if img is None:
                    print(f"无法读取图像: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
                
                # 创建绘图
                fig, ax = plt.subplots(1, figsize=(12, 8))
                ax.imshow(img)
                
                # 添加图例条目
                legend_handles = []
                
                # 处理每组标签
                for label_idx, label_source in enumerate(labels_dict):
                    label_path = label_source[f'data{data_class}']
                    
                    # 解析标签文件
                    boxes = parse_labels(label_path, img_name)
                    
                    color = colors[label_idx]
                    
                    # 绘制边界框
                    for box in boxes:
                        if len(box) >= 5:  # 确保box格式正确
                            # 添加到图像上
                            x, y, w, h = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                            
                            # 转换归一化坐标到像素坐标
                            height, width, _ = img.shape
                            x_center = x * width
                            y_center = y * height
                            box_width = w * width
                            box_height = h * height
                            
                            # 创建矩形
                            rect = patches.Rectangle(
                                (x_center - box_width/2, y_center - box_height/2),
                                box_width, box_height,
                                linewidth=2,
                                edgecolor=color,
                                facecolor='none'
                            )
                            ax.add_patch(rect)
                    
                    # 添加图例
                    legend_handles.append(patches.Patch(color=color, label=label_source['name']))
                
                # 显示图例
                ax.legend(handles=legend_handles, loc='upper left')
                
                # 关闭坐标轴
                ax.axis('off')
                
                # 保存图像到对应数据集的文件夹
                save_path = os.path.join(output_dirs[data_class], f"{img_name}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                
                print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # 可以通过以下方式直接调用
    labels_list = [
        '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/medical/v8',
        
    ]
    base_output_dir = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/medical/compare0612"
    csv_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/medical/compare0612/labels_comparison.csv"
    
    main(labels_list=labels_list, base_output_dir=base_output_dir, csv_path=csv_path)
    
    # 或者不传参数，使用命令行参数
    # main()