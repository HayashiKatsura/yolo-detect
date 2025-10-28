import pandas as pd
import os
import glob
import csv

def mergescv(csv_list: list, save_path: str):
    """
    合并多个列数相同的CSV文件为一个CSV文件。

    参数:
        csv_list (list): 要合并的CSV文件路径列表。
        save_path (str): 合并后保存的CSV文件路径。
    """
    df_list = []

    for csv_file in csv_list:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"读取文件 {csv_file} 时出错：{e}")

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"合并完成，保存至：{save_path}")
    else:
        print("未成功读取任何CSV文件，未生成输出。")

def convert_txt_to_csv(input_dir, output_csv):
    """
    将输入目录中的所有.txt文件转换为单个CSV文件
    表头为: file_name, x, y, w, h, conf
    注意：每个txt文件行中的第一个值会被忽略，只取后5个值
    """
    # 创建/打开输出的CSV文件
    if os.path.exists(output_csv):
        os.remove(output_csv)
    with open(output_csv, 'w', newline='') as csvfile:
        # 创建CSV写入器并设置表头
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_name', 'x', 'y', 'w', 'h', 'conf'])
        
        # 查找目录中的所有.txt文件
        txt_files = list(sorted(glob.glob(os.path.join(input_dir, '*.txt'))))
        
        # 处理每个.txt文件
        for txt_file in txt_files:
            # 获取不带扩展名的文件名
            file_name = os.path.basename(txt_file)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # 读取.txt文件的内容
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
                # 处理文件中的每一行
                for line in lines:
                    # 按空格分割行并去除空白
                    values = line.strip().split()
                    values = [round(float(v),2) for v in values]
                    
                    # 确保至少有6个值
                    if len(values) >= 6:
                        # 忽略第一个值，只取后5个值，写入行: file_name, x, y, w, h, conf
                        csv_writer.writerow([file_name_without_ext] + values[1:6])
                    else:
                        print(f"警告: 跳过{file_name}中的一行，因为它没有至少6个值。")

def save_csv(data: list, save_path: str) -> None:
    # 先进行排序，先按 conf 升序，再按 filesName 升序
    sorted_data = sorted(data, key=lambda x: (x['labels'], x['filesName']))
    
    # 写入 CSV 文件
    with open(save_path, mode='w', newline='') as csvfile:
        fieldnames = ['filesName','labels','conf','min_conf','max_conf','count','threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()

        # 写入数据
        writer.writerows(sorted_data)

    print(f"Results have been saved to {save_path}")

import os
import csv
from pathlib import Path

def yolo_to_pixel(center_x, center_y, width, height, img_width=1000, img_height=1000):
    """
    将YOLO格式的中心点坐标转换为像素坐标
    YOLO格式: (center_x, center_y, width, height) 都是相对于图像尺寸的比例 (0-1)
    返回: (x_min, y_min, x_max, y_max) 像素坐标
    """
    # 转换为像素坐标
    center_x_pixel = center_x * img_width
    center_y_pixel = center_y * img_height
    width_pixel = width * img_width
    height_pixel = height * img_height
    
    # 计算左上角和右下角坐标
    x_min = int(center_x_pixel - width_pixel / 2)
    y_min = int(center_y_pixel - height_pixel / 2)
    x_max = int(center_x_pixel + width_pixel / 2)
    y_max = int(center_y_pixel + height_pixel / 2)
    
    return x_min, y_min, x_max, y_max

def read_yolo_file(file_path):
    """
    读取YOLO格式的标注文件
    返回标注列表，每个标注包含 [class_id, x_min, y_min, x_max, y_max]
    """
    annotations = []
    if not os.path.exists(file_path):
        return annotations
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 转换为像素坐标
                    x_min, y_min, x_max, y_max = yolo_to_pixel(center_x, center_y, width, height)
                    annotations.append([class_id, x_min, y_min, x_max, y_max])
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return annotations

def convert_labels_to_csv(labels_folder_1, labels_folder_2, save_path):
    """
    将两个标注文件夹中的YOLO格式标注转换为CSV格式
    """
    # 获取所有txt文件
    folder1_files = set()
    folder2_files = set()
    
    if os.path.exists(labels_folder_1):
        folder1_files = {f for f in os.listdir(labels_folder_1) if f.endswith('.txt')}
    
    if os.path.exists(labels_folder_2):
        folder2_files = {f for f in os.listdir(labels_folder_2) if f.endswith('.txt')}
    
    # 获取所有文件的并集
    all_files = folder1_files.union(folder2_files)
    
    # CSV表头
    header = ['file_name', 'type_class_1', 'x_min_1', 'y_min_1', 'x_max_1', 'y_max_1',
              'type_class_2', 'x_min_2', 'y_min_2', 'x_max_2', 'y_max_2']
    
    # 准备CSV数据
    csv_data = []
    
    for filename in sorted(all_files):
        # 读取两个文件夹中的标注
        file1_path = os.path.join(labels_folder_1, filename) if labels_folder_1 else ""
        file2_path = os.path.join(labels_folder_2, filename) if labels_folder_2 else ""
        
        annotations1 = read_yolo_file(file1_path)
        annotations2 = read_yolo_file(file2_path)
        
        # 确定最大行数
        max_rows = max(len(annotations1), len(annotations2))
        
        # 如果两个文件都没有标注，跳过
        if max_rows == 0:
            continue
        
        # 生成CSV行
        for i in range(max_rows):
            row = [filename]  # file_name
            
            # 添加第一个文件夹的标注
            if i < len(annotations1):
                row.extend(annotations1[i])  # type_class_1, x_min_1, y_min_1, x_max_1, y_max_1
            else:
                row.extend(['', '', '', '', ''])  # 空值
            
            # 添加第二个文件夹的标注
            if i < len(annotations2):
                row.extend(annotations2[i])  # type_class_2, x_min_2, y_min_2, x_max_2, y_max_2
            else:
                row.extend(['', '', '', '', ''])  # 空值
            
            csv_data.append(row)
    
    # 写入CSV文件
    csv_file_path = save_path if save_path.endswith('.csv') else f"{save_path}.csv"
    
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(csv_data)
        
        print(f"成功将标注转换为CSV格式，保存到: {csv_file_path}")
        print(f"共处理了 {len(all_files)} 个文件，生成了 {len(csv_data)} 行数据")
        
    except Exception as e:
        print(f"写入CSV文件时出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 设置参数
    labels_folder_1 = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/labels/train"  # 第一个标注文件夹路径
    labels_folder_2 = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/predict/train/train_predict_results_conf0.25"  # 第二个标注文件夹路径
    save_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/predict/train"  # 输出CSV文件路径（不需要.csv后缀）
    
    # 执行转换
    convert_labels_to_csv(labels_folder_1, labels_folder_2, save_path)