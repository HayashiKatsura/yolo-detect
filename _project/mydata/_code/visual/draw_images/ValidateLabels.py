import cv2
import os
import random

def validate_label_to_image(
                image_path,
                label_path,
                colors_classes_map,
                save_path):
    """
    将标注文件转换为可视化图像
    Args:
        image_path (_type_): 图像路径
        label_path (_type_): 标注文件路径
        colors_classes_map (_type_): 类别颜色映射
        save_path (_type_): 保存路径
    
    示例：    
    colors_classes_map = {
    0: ((0, 0, 0),'black'),         # black
    1: ((0, 0, 255),'damage'),       # damage - red
    2: ((255, 0, 0),'ink'),       # ink - blue
    3: ((0, 255, 0),'resuide'),       # resuide - green
    4: ((0, 255, 255),'pi'),     # pi - yellow
    5: ((255, 0, 255),'circle'),     # circle - purple
    }

    Raises:
        ValueError: _description_
        FileNotFoundError: _description_
    """
    
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    height, width = image.shape[:2]

    # 2. 读取标注
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标注文件不存在：{label_path}")
    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # 不合法格式
        if len(parts) == 5:
            parts.append('1.0')
        cls_id, cx, cy, w, h, conf = parts
        cls_id = int(cls_id)
        cx, cy, w, h, conf = map(float, [cx, cy, w, h, conf])

        # YOLO 转为左上角坐标
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)


        color = colors_classes_map.get(cls_id)[0]
        label = f"{colors_classes_map.get(cls_id)[1]} {conf:.2f}"

        # 画框和文字
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. 保存图像
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), image)
    print(f"已保存标注图像至：{save_path}")


def batch_validate_labels_to_images(
                image_folder,
                label_folder,
                colors_classes_map,
                save_path):
    """
    批量转换标注文件为可视化图像
    Args:
        image_folder (_type_): _description_
        label_folder (_type_): _description_
        colors_classes_map (_type_): _description_
        save_path (_type_): _description_
    """
    cnt = 0
    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            label_path = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
            # save_path = os.path.join(save_path, img_file)
            validate_label_to_image(
                image_path=img_path,
                label_path=label_path,
                colors_classes_map=colors_classes_map,
                save_path=save_path
            )
            cnt += 1
    print(f"已处理{cnt}张图像")
            


# def validate_multi_labels_to_image(
#     image_path,
#     label_path,
#     colors_classes_map,
#     save_path,
#     colors_list=None,
#     name_list=None
# ):
#     """
#     可视化一个或多个标签文件，并在图像左上角添加图例。

#     Args:
#         image_path (str): 原始图像路径
#         label_path (str or list): 一个或多个标签路径,具体的路径，不是文件夹
#         colors_classes_map (dict): 类别颜色映射（用于获取类别名）
#         save_path (str): 可视化图像保存路径
#         colors_list (list): 每组标签使用的颜色（强制统一）
#         name_list (list): 每组标签颜色的说明名称（用于图例）
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"无法读取图像：{image_path}")
#     height, width = image.shape[:2]

#     # 统一 label_path 为列表
#     if not isinstance(label_path, list):
#         label_path = [label_path]

#     # 自动生成颜色
#     if colors_list is None:
#         colors_list = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(label_path))]

#     # 默认图例名称
#     if name_list is None:
#         name_list = [f"Label Group {i+1}" for i in range(len(label_path))]

#     if len(label_path) != len(colors_list) or len(label_path) != len(name_list):
#         raise ValueError("label_path、colors_list 和 name_list 的长度必须一致")

#     for i, single_label_path in enumerate(label_path):
#         if not os.path.exists(single_label_path):
#             print(f"警告：未找到标签文件 {single_label_path}，跳过。")
#             continue

#         with open(single_label_path, 'r') as f:
#             lines = f.readlines()

#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) != 6:
#                 continue
#             cls_id, cx, cy, w, h, conf = parts
#             cls_id = int(cls_id)
#             cx, cy, w, h, conf = map(float, [cx, cy, w, h, conf])

#             x1 = int((cx - w / 2) * width)
#             y1 = int((cy - h / 2) * height)
#             x2 = int((cx + w / 2) * width)
#             y2 = int((cy + h / 2) * height)

#             color = colors_list[i]
#             label_name = colors_classes_map.get(cls_id, ('unknown', 'unknown'))[1]
#             label = f"{label_name} {conf:.2f}"

#             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(image, label, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # 绘制图例
#     legend_x, legend_y = 10, 20
#     for i, name in enumerate(name_list):
#         color = colors_list[i]
#         cv2.rectangle(image, (legend_x, legend_y - 12), (legend_x + 20, legend_y + 5), color, -1)
#         cv2.putText(image, name, (legend_x + 25, legend_y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#         legend_y += 25

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), image)
#     print(f"已保存标注图像至：{save_path}")

    
# def batch_validate_multi_labels_to_images(
#     images_folder,
#     label_folder,
#     colors_classes_map,
#     save_path,
#     colors_list=None,
#     name_list=None
# ):
#     """
#     批量可视化多个标签文件，并在图像左上角添加图例。
#     Args:
#         images_folder (str): 原始图像文件夹路径
#         labels_folder (str): 多个标签文件夹路径
#         colors_classes_map (dict): 类别颜色映射（用于获取类别名）
#         save_folder (str): 可视化图像保存文件夹路径
#         colors_list (list): 每组标签使用的颜色（强制统一）
#         name_list (list): 每组标签颜色的说明名称（用于图例）
#     """
#     cnt = 0
#     for img_file in os.listdir(images_folder):
#         if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             labels = []
#             for label_items in label_folder:
#                 this_label_path = os.path.join(label_items, os.path.splitext(img_file)[0] + '.txt')
#                 if os.path.exists(this_label_path):
#                     labels.append(this_label_path)
#             validate_multi_labels_to_image(
#                 image_path=os.path.join(images_folder, img_file),
#                 label_path=labels,
#                 colors_classes_map=colors_classes_map,
#                 save_path=save_path,
#                 colors_list=colors_list,
#                 name_list=name_list
#             )
#             cnt += 1
#     print(f"已处理{cnt}张图像")


import cv2
import os
import random
from typing import List, Dict, Tuple, Union, Optional


def visualize_multi_labels_on_image(
    image_path: str,
    label_paths: Union[str, List[str]],
    class_names_map: Dict[int, str],
    output_path: str,
    colors_list: Optional[List[Tuple[int, int, int]]] = None,
    legend_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.0
) -> bool:
    """
    在单张图像上可视化一个或多个标签文件，并在图像左上角添加图例。

    Args:
        image_path (str): 原始图像路径
        label_paths (str or List[str]): 一个或多个标签文件路径
        class_names_map (Dict[int, str]): 类别ID到类别名称的映射
        output_path (str): 输出图像的完整路径（包含文件名）
        colors_list (List[Tuple[int, int, int]], optional): 每组标签使用的颜色 (B, G, R)
        legend_names (List[str], optional): 每组标签在图例中的名称
        confidence_threshold (float): 置信度阈值，低于此值的框不显示

    Returns:
        bool: 处理成功返回True，失败返回False
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return False
    
    height, width = image.shape[:2]

    # 统一处理label_paths为列表
    if isinstance(label_paths, str):
        label_paths = [label_paths]

    # 过滤存在的标签文件
    valid_label_paths = []
    for label_path in label_paths:
        if os.path.exists(label_path):
            valid_label_paths.append(label_path)
        else:
            print(f"警告：标签文件不存在 {label_path}")
    
    if not valid_label_paths:
        print("错误：没有找到有效的标签文件")
        return False

    num_labels = len(valid_label_paths)

    # 生成颜色
    if colors_list is None:
        colors_list = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(num_labels)
        ]
    elif len(colors_list) < num_labels:
        print(f"警告：提供的颜色数量({len(colors_list)})少于标签文件数量({num_labels})")
        # 扩展颜色列表
        while len(colors_list) < num_labels:
            colors_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # 生成图例名称
    if legend_names is None:
        legend_names = [f"标签组 {i+1}" for i in range(num_labels)]
    elif len(legend_names) < num_labels:
        print(f"警告：提供的图例名称数量({len(legend_names)})少于标签文件数量({num_labels})")
        # 扩展图例名称
        while len(legend_names) < num_labels:
            legend_names.append(f"标签组 {len(legend_names)+1}")

    # 处理每个标签文件
    for i, label_path in enumerate(valid_label_paths):
        color = colors_list[i]
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"错误：无法读取标签文件 {label_path}: {e}")
            continue

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 5:
                print(f"警告：{label_path} 第{line_num}行格式不正确，跳过")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # 检查是否有置信度信息
                conf = float(parts[5]) if len(parts) > 5 else 1.0
                
                # 置信度过滤
                if conf < confidence_threshold:
                    continue

                # 转换为像素坐标
                x1 = max(0, int((cx - w / 2) * width))
                y1 = max(0, int((cy - h / 2) * height))
                x2 = min(width, int((cx + w / 2) * width))
                y2 = min(height, int((cy + h / 2) * height))

                # 获取类别名称
                class_name = class_names_map.get(cls_id, f"类别_{cls_id}")
                
                # 构建标签文本
                if len(parts) > 5:  # 有置信度信息
                    label_text = f"{class_name} {conf:.2f}"
                else:
                    label_text = class_name

                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签文本背景
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(image, (x1, y1 - text_height - 5), 
                            (x1 + text_width, y1), color, -1)
                
                # 绘制标签文本
                cv2.putText(image, label_text, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except (ValueError, IndexError) as e:
                print(f"警告：{label_path} 第{line_num}行数据解析错误: {e}")
                continue

    # 绘制图例
    _draw_legend(image, legend_names, colors_list)

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    success = cv2.imwrite(output_path, image)
    if success:
        print(f"成功保存标注图像至：{output_path}")
        return True
    else:
        print(f"错误：保存图像失败 {output_path}")
        return False


def batch_visualize_multi_labels(
    images_folder: str,
    label_folders: List[str],
    class_names_map: Dict[int, str],
    output_folder: str,
    colors_list: Optional[List[Tuple[int, int, int]]] = None,
    legend_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.0,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> int:
    """
    批量可视化多个标签文件夹中的标签。

    Args:
        images_folder (str): 原始图像文件夹路径
        label_folders (List[str]): 多个标签文件夹路径列表
        class_names_map (Dict[int, str]): 类别ID到类别名称的映射
        output_folder (str): 可视化图像保存文件夹路径
        colors_list (List[Tuple[int, int, int]], optional): 每组标签使用的颜色
        legend_names (List[str], optional): 每组标签在图例中的名称
        confidence_threshold (float): 置信度阈值
        image_extensions (Tuple[str, ...]): 支持的图像文件扩展名

    Returns:
        int: 成功处理的图像数量
    """
    if not os.path.exists(images_folder):
        print(f"错误：图像文件夹不存在 {images_folder}")
        return 0

    # 检查标签文件夹
    valid_label_folders = []
    for folder in label_folders:
        if os.path.exists(folder):
            valid_label_folders.append(folder)
        else:
            print(f"警告：标签文件夹不存在 {folder}")

    if not valid_label_folders:
        print("错误：没有找到有效的标签文件夹")
        return 0

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(image_extensions)]

    print(f"找到 {len(image_files)} 张图像，开始处理...")

    for img_file in image_files:
        # 构建对应的标签文件路径
        base_name = os.path.splitext(img_file)[0]
        label_paths = []
        
        for label_folder in valid_label_folders:
            label_path = os.path.join(label_folder, base_name + '.txt')
            if os.path.exists(label_path):
                label_paths.append(label_path)

        if not label_paths:
            print(f"警告：图像 {img_file} 没有找到对应的标签文件")
            continue

        # 构建输出路径
        image_path = os.path.join(images_folder, img_file)
        output_path = os.path.join(output_folder, img_file)

        # 处理单张图像
        if visualize_multi_labels_on_image(
            image_path=image_path,
            label_paths=label_paths,
            class_names_map=class_names_map,
            output_path=output_path,
            colors_list=colors_list,
            legend_names=legend_names,
            confidence_threshold=confidence_threshold
        ):
            processed_count += 1

    print(f"批量处理完成，成功处理 {processed_count} 张图像")
    return processed_count

import numpy
def _draw_legend(
    image: 'numpy.ndarray', 
    legend_names: List[str], 
    colors_list: List[Tuple[int, int, int]]
) -> None:
    """
    在图像左上角绘制图例。

    Args:
        image: OpenCV图像数组
        legend_names: 图例名称列表
        colors_list: 颜色列表
    """
    legend_x, legend_y = 10, 30
    legend_rect_size = 15
    legend_spacing = 25

    for i, (name, color) in enumerate(zip(legend_names, colors_list)):
        # 绘制颜色方块
        cv2.rectangle(
            image, 
            (legend_x, legend_y - legend_rect_size), 
            (legend_x + legend_rect_size, legend_y), 
            color, 
            -1
        )
        
        # 绘制边框
        cv2.rectangle(
            image, 
            (legend_x, legend_y - legend_rect_size), 
            (legend_x + legend_rect_size, legend_y), 
            (0, 0, 0), 
            1
        )
        
        # 绘制文本
        cv2.putText(
            image, 
            name, 
            (legend_x + legend_rect_size + 5, legend_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )
        
        # 绘制文本轮廓（提高可见性）
        cv2.putText(
            image, 
            name, 
            (legend_x + legend_rect_size + 5, legend_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
        
        legend_y += legend_spacing

# 使用示例
if __name__ == "__main__":
    # 定义类别映射
    class_names = {
        0: "black",
        1: "damage", 
        2: "ink",
        3: "resuide",
        4: "pi",
        5: "circle"
    }
    
    # 单张图像可视化示例
    # success = visualize_multi_labels_on_image(
    #     image_path="path/to/image.jpg",
    #     label_paths=["path/to/labels1.txt", "path/to/labels2.txt"],
    #     class_names_map=class_names,
    #     output_path="path/to/output/result.jpg",
    #     colors_list=[(0, 255, 0), (255, 0, 0)],  # 绿色，红色
    #     legend_names=["模型1", "模型2"]
    # )
    
    # 批量处理示例
    processed_count = batch_visualize_multi_labels(
        images_folder="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/images/val",
        label_folders=["/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/labels/val"],
        class_names_map=class_names,
        output_folder="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/validate",
        colors_list=[(255, 0, 0)],
        legend_names=["true-labels"],
        confidence_threshold=0.25
    )

