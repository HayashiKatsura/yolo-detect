import cv2
import numpy as np
def concat_images(extracted_image, background_image_path, transparency=None, scale_factor=0.2):
    """
    将抠出的异常样本和正常的样本叠加到一起
    
    参数:
    extracted_image: 抠出的彩色图像
    background_image_path: 背景图片的文件路径
    transparency: 透明度，值范围0-1，1表示完全不透明，None表示完全不透明
    scale_factor: 抠出图像相对于背景图像的缩放比例，默认为0.2
    
    返回:
    (result, overlay_mask): 叠加后的图像和对应的掩码（黑色背景，提取区域为白色）
    """
    # 读取背景图片
    background = cv2.imread(background_image_path)
    if background is None:
        raise ValueError(f"无法读取背景图片: {background_image_path}")
    
    # 如果背景图片是BGR格式，转换为RGB
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    # 确保extracted_image是RGB格式
    if len(extracted_image.shape) == 2:
        extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_GRAY2RGB)
    elif extracted_image.shape[2] == 4:  # 如果是RGBA格式
        extracted_image = extracted_image[:, :, :3]
    
    # # 将 extracted_image 转为灰度图
    # _extracted_image = cv2.cvtColor(extracted_image,cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('_extracted_image.png',_extracted_image)
    # 创建一个掩码，白色区域(255)表示提取的区域，黑色区域(0)表示背景
    mask = np.all(extracted_image != 255, axis=2).astype(np.uint8) * 255
    # cv2.imwrite('_mask.png',mask)
    # 背景图像的尺寸
    bg_height, bg_width = background.shape[:2]
    
    # 异常样本的尺寸
    # 根据scale_factor计算，同时确保不超过背景图像的尺寸
    extracted_height, extracted_width = extracted_image.shape[:2]
    
    # 计算缩放后的宽高
    target_width = int(bg_width * scale_factor)
    target_height = int(bg_height * scale_factor)
    
    # 保持原始宽高比
    orig_aspect_ratio = extracted_width / extracted_height
    if target_width / target_height > orig_aspect_ratio:
        # 高度限制
        target_width = int(target_height * orig_aspect_ratio)
    else:
        # 宽度限制
        target_height = int(target_width / orig_aspect_ratio)
    
    # 确保不超过背景图像尺寸
    target_width = min(target_width, bg_width)
    target_height = min(target_height, bg_height)
    
    # 调整抠出图像和掩码的大小
    extracted_image_resized = cv2.resize(extracted_image, (target_width, target_height))
    # mask_resized = cv2.resize(mask, (target_width, target_height))
    mask_resized = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    
    # 创建一个结果图像的副本
    result = background.copy()
    
    # 创建一个与背景相同大小的全黑掩码
    overlay_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    
    # 计算抠出图像在背景中的位置（居中）
    y_offset = (bg_height - target_height) // 2
    x_offset = (bg_width - target_width) // 2
    
    # 获取抠出图像的区域
    roi = result[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
    
    # 将调整大小后的掩码放置在背景掩码的对应位置
    overlay_mask[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = mask_resized
    
    # 如果设置了透明度
    if transparency is not None:
        alpha = transparency
        # 仅在mask区域应用透明度混合
        for c in range(3):  # 对RGB三个通道分别处理
            roi[:, :, c] = np.where(
                mask_resized > 0,
                roi[:, :, c] * (1 - alpha) + extracted_image_resized[:, :, c] * alpha,
                roi[:, :, c]
            )
    else:
        # 直接硬覆盖
        for c in range(3):
            roi[:, :, c] = np.where(
                mask_resized > 0,
                extracted_image_resized[:, :, c],
                roi[:, :, c]
            )
    
    return result, overlay_mask

def concat_images_with_positions(extracted_image, background_image_path, transparency=None, scale_factor=0.2):
    """
    将抠出的异常样本和正常的样本叠加到一起

    
    参数:
    extracted_image: 抠出的彩色图像
    background_image_path
    transparency: 透明度，值范围0-1，1表示完全不透明，None表示完全不透明
    scale_factor: 抠出图像相对于背景图像的缩放比例，默认为0.2
    
    返回:
    (result, overlay_mask, yolo_positions): 
        - result: 叠加后的图像
        - overlay_mask: 对应的掩码（黑色背景，提取区域为白色）
        - yolo_positions: 列表，包含所有区域的YOLO格式位置信息 [x_center, y_center, width, height]（归一化值）
    """
    # 读取背景图片
    background = cv2.imread(background_image_path)
    if background is None:
        raise ValueError(f"无法读取背景图片: {background_image_path}")
    
    # 如果背景图片是BGR格式，转换为RGB
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    # 确保extracted_image是RGB格式
    if len(extracted_image.shape) == 2:
        extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_GRAY2RGB)
    elif extracted_image.shape[2] == 4:  # 如果是RGBA格式
        extracted_image = extracted_image[:, :, :3]
    
    # cv2.imwrite('extracted_image.png',extracted_image)
    
    # 创建一个掩码，白色区域(255)表示提取的区域，黑色区域(0)表示背景
    mask = np.all(extracted_image != 255, axis=2).astype(np.uint8) * 255
    # 使用形态学操作填充内部破损
    kernel = np.ones((3, 3), np.uint8)  # 3x3膨胀核
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite('mask.png',mask)
    # 计算背景图像的尺寸
    bg_height, bg_width = background.shape[:2]
    
    # 异常样本的尺寸
    # 根据scale_factor计算，同时确保不超过背景图像的尺寸
    extracted_height, extracted_width = extracted_image.shape[:2]
    
    # 计算缩放后的宽高
    target_width = int(bg_width * scale_factor)
    target_height = int(bg_height * scale_factor)
    
    # 保持原始宽高比
    orig_aspect_ratio = extracted_width / extracted_height
    if target_width / target_height > orig_aspect_ratio:
        # 高度限制
        target_width = int(target_height * orig_aspect_ratio)
    else:
        # 宽度限制
        target_height = int(target_width / orig_aspect_ratio)
    
    # 确保不超过背景图像尺寸
    target_width = min(target_width, bg_width)
    target_height = min(target_height, bg_height)
    
    # 调整抠出图像和掩码的大小
    extracted_image_resized = cv2.resize(extracted_image, (target_width, target_height))
    mask_resized = cv2.resize(mask, (target_width, target_height))
    
    # 计算抠出图像在背景中的位置（居中）
    y_offset = (bg_height - target_height) // 2
    x_offset = (bg_width - target_width) // 2
    
    # 创建一个结果图像的副本
    result = background.copy()
    
    # 创建一个与背景相同大小的全黑掩码
    overlay_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    
    # 获取抠出图像的区域
    roi = result[y_offset:y_offset+target_height, x_offset:x_offset+target_width]
    
    # 将调整大小后的掩码放置在背景掩码的对应位置
    overlay_mask[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = mask_resized
    
    # 如果设置了透明度
    if transparency is not None:
        alpha = transparency
        # 仅在mask区域应用透明度混合
        for c in range(3):  # 对RGB三个通道分别处理
            roi[:, :, c] = np.where(
                mask_resized > 0,
                roi[:, :, c] * (1 - alpha) + extracted_image_resized[:, :, c] * alpha,
                roi[:, :, c]
            )
    else:
        # 直接硬覆盖
        for c in range(3):
            roi[:, :, c] = np.where(
                mask_resized > 0,
                extracted_image_resized[:, :, c],
                roi[:, :, c]
            )
    
    # 查找叠加后掩码中的所有连通区域
    # cv2.imwrite('overlay_mask.png',overlay_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(overlay_mask, connectivity=8)
    """
    num_labels：找到的连通区域数量（包括背景，背景通常标记为0）
    labels：标记矩阵，与输入图像大小相同，每个像素值表示它所属的连通区域的标签
    
    stats：统计信息矩阵，对于每个标签（包括背景），包含以下统计信息：
    左上角x坐标（CV_CC_STAT_LEFT）
    左上角y坐标（CV_CC_STAT_TOP）
    宽度（CV_CC_STAT_WIDTH）
    高度（CV_CC_STAT_HEIGHT）
    区域面积（CV_CC_STAT_AREA）
    
    centroids：质心坐标，每个连通区域的中心点(x,y)坐标

    """
    
    # # TODO ↓
    # # 创建彩色图像来显示所有连通区域
    # # 使用随机颜色标记每个连通区域
    # # 创建RGB图像
    # colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    # # 为每个标签生成随机颜色
    # colors = []
    # # 背景通常设为黑色
    # colors.append([0, 0, 0])  
    # import random
    # # 为每个连通区域生成随机颜色
    # for i in range(1, num_labels):
    #     colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    # # 将每个连通区域用不同颜色填充
    # for i in range(0, num_labels):
    #     colored_labels[labels == i] = colors[i]


    # # 可选：在图像上绘制每个连通区域的中心点和标签编号
    # result_with_info = colored_labels.copy()
    # for i in range(1, num_labels):
    #     # 绘制中心点
    #     cv2.circle(result_with_info, (int(centroids[i][0]), int(centroids[i][1])), 5, (255, 255, 255), -1)
    #     # 添加标签编号
    #     # cv2.putText(result_with_info, str(i), (int(centroids[i][0]) + 10, int(centroids[i][1])), 
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # cv2.imwrite('result_with_info.png',result_with_info)
    # # ↑
    # print()
    
    
    
    # YOLO位置信息列表（不包括背景，背景标签为0）
    yolo_positions = []
    
    # 从标签1开始（跳过背景标签0）
    for i in range(1, num_labels):
        # 获取当前连通区域的位置和大小
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 计算YOLO格式的归一化位置信息
        x_center = (x + w / 2) / bg_width
        y_center = (y + h / 2) / bg_height
        width = w / bg_width
        height = h / bg_height
        
        # 添加到列表
        yolo_positions.append([x_center, y_center, width, height])
    
    # 过滤掉最后一位小数小于 0.01 的子列表
    yolo_positions = [pos for pos in yolo_positions if pos[-1] >= 0.01]

    
    return result, overlay_mask, yolo_positions

import os
from PIL import Image, ImageDraw, ImageFont
import os
from concurrent.futures import ThreadPoolExecutor
import gc


def process_images(source_folder_base: str, filename_list: dict, save_folder_base: str, n: int):
    """
    优化后的处理函数，主要改进字体加载和多线程处理。
    """
    # 加载字体，整个函数只加载一次
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for key, value_list in filename_list.items():
        current_source_folder = source_folder_base
        current_save_folder = os.path.join(save_folder_base, 'visualize')

        if key == 'wrong_check':
            current_source_folder = os.path.join(current_source_folder, '_normally')
            current_save_folder = os.path.join(current_save_folder, 'wrong_check')
        elif key == 'miss_check':
            current_source_folder = os.path.join(current_source_folder, '_anomaly')
            current_save_folder = os.path.join(current_save_folder, 'miss_check')

        os.makedirs(current_save_folder, exist_ok=True)

        processed_images = []

        def process_single_file(filename):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                print(f"警告: 文件 '{filename}' 不是图片文件，已跳过。")
                return None
            try:
                file_path = os.path.join(current_source_folder, filename)
                img = Image.open(file_path)
                # 使用更快的采样方法
                resized_img = img.resize((256, 256), resample=Image.Resampling.NEAREST)
                padded_img = Image.new('RGB', (256, 276), color='white')
                padded_img.paste(resized_img, (0, 20))
                draw = ImageDraw.Draw(padded_img)
                draw.text((2, 2), os.path.splitext(filename)[0], fill='black', font=font)
                return (os.path.splitext(filename)[0], padded_img)
            except Exception as e:
                print(f"处理文件 '{filename}' 时发生错误: {e}")
                return None

        # 使用多线程处理文件
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_single_file, value_list)
            for result in results:
                if result:
                    processed_images.append(result)

        # 后续合并和保存逻辑保持不变
        if not processed_images:
            print(f"在类别 '{key}' 下没有找到或处理任何图片文件。")
            continue

        image_groups = [processed_images[i:i + n * 3] for i in range(0, len(processed_images), n * 3)]

        for group_index, image_group in enumerate(image_groups):
            if not image_group:
                continue
            rows = min(3, (len(image_group) + n - 1) // n)
            images_per_row = min(n, len(image_group))
            combined_width = images_per_row * 256 + (images_per_row - 1) * 20
            combined_height = rows * 276 + (rows - 1) * 20
            combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

            x_offset, y_offset = 0, 0
            for i, (filename, img) in enumerate(image_group):
                combined_image.paste(img, (x_offset, y_offset))
                x_offset += 256 + 20
                if (i + 1) % n == 0 or i + 1 == len(image_group):
                    y_offset += 276 + 20
                    x_offset = 0

            output_filename = f"{'_'.join([fn for fn, _ in image_group])}_merged_{group_index + 1}.png"
            output_path = os.path.join(current_save_folder, output_filename)
            combined_image.save(output_path, compress_level=1)  # 降低压缩级别以加快保存
            print(f"已保存合并后的图像: {output_path}")
    gc.collect()
    
if __name__ == "__main__":

    
    source_folder_base = "/kweilx/ultralytics/_project/mydata/_results/yolo11m/single_class/single_anomaly/synthetic_train210_val90_202503171648/_predict"  # 替换为您的源文件夹基础路径
    filename_list = {
        "wrong_check":
        [

        ],
       "miss_check":
            [
"065.png",
"069.png"  
        ]
    }
    save_folder_base = os.path.join(source_folder_base, "merged_images")  # 替换为您希望保存合并后图像的基础文件夹路径
    n = 3  # 每行显示的图片数量
    process_images(source_folder_base, filename_list, save_folder_base, n)