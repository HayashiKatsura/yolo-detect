import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

def merge_all_images(source_list: List[str], 
                    name_list: Optional[List[str]] = None,
                    resize: int = 200,
                    gap: int = 10,
                    output_dir: str = "merged_images"):
    """
    遍历文件夹下的所有图像文件，将同名图像合并并保存到指定目录
    
    参数:
    source_list (List[str]): 文件夹路径列表
    name_list (List[str], optional): 写在每幅图下方的文件名，默认为当前文件名
    resize (int): 放缩到指定大小，等比例缩放
    gap (int): 图像间距，默认10像素
    output_dir (str): 输出目录路径
    """
    
    # 验证参数
    if not source_list:
        raise ValueError("source_list不能为空")
    
    if name_list and len(name_list) != len(source_list):
        raise ValueError("name_list的长度必须与source_list一致")
    
    # 如果没有提供name_list，使用文件夹名称
    if name_list is None:
        name_list = [os.path.basename(path.rstrip(os.sep)) for path in source_list]
    
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 获取所有文件夹中的图像文件
    all_images = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    for folder_path in source_list:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在")
            continue
            
        files = os.listdir(folder_path)
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        
        for img_file in image_files:
            if img_file not in all_images:
                all_images[img_file] = []
            all_images[img_file].append(folder_path)
    
    # 找到在所有文件夹中都存在的图像文件
    common_images = [img for img, folders in all_images.items() 
                    if len(folders) == len(source_list)]
    
    if not common_images:
        raise ValueError("没有找到在所有文件夹中都存在的图像文件")
    
    print(f"找到 {len(common_images)} 个图像文件需要处理")
    
    # 设置字体（尝试使用系统字体，如果失败则使用默认字体）
    try:
        font_size = max(12, resize // 15)
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 处理每个图像文件
    success_count = 0
    error_count = 0
    
    for img_idx, selected_image in enumerate(common_images):
        try:
            print(f"处理第 {img_idx + 1}/{len(common_images)} 个文件: {selected_image}")
            
            # 读取和处理图像
            images = []
            for i, folder_path in enumerate(source_list):
                img_path = os.path.join(folder_path, selected_image)
                img = Image.open(img_path)
                
                # 转换为RGB模式（如果需要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 等比例缩放
                width, height = img.size
                if width > height:
                    new_width = resize
                    new_height = int(height * (resize / width))
                else:
                    new_height = resize
                    new_width = int(width * (resize / height))
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                images.append((img, name_list[i]))
            
            # 计算合并后图像的尺寸
            max_height = max(img.size[1] for img, _ in images)
            text_height = 30  # 为文本预留的高度
            total_width = sum(img.size[0] for img, _ in images) + gap * (len(images) - 1)
            total_height = max_height + text_height
            
            # 创建新图像
            merged_img = Image.new('RGB', (total_width, total_height), 'white')
            draw = ImageDraw.Draw(merged_img)
            
            # 粘贴图像和添加文本
            x_offset = 0
            for img, name in images:
                # 计算垂直居中位置
                y_offset = (max_height - img.size[1]) // 2
                
                # 粘贴图像
                merged_img.paste(img, (x_offset, y_offset))
                
                # 添加文本
                bbox = draw.textbbox((0, 0), name, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = x_offset + (img.size[0] - text_width) // 2
                text_y = max_height + 5
                
                draw.text((text_x, text_y), name, fill='black', font=font)
                
                # 更新x偏移
                x_offset += img.size[0] + gap
            
            # 保存结果，保持原始文件名
            filename, ext = os.path.splitext(selected_image)
            output_filename = f"{filename}_merged{ext}"
            output_path = os.path.join(output_dir, output_filename)
            merged_img.save(output_path, quality=95)
            
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {selected_image} 时出错: {str(e)}")
            error_count += 1
            continue
    
    print(f"\n处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"结果保存在: {output_dir}")
    
    return success_count, error_count

# 使用示例
if __name__ == "__main__":
    # 示例用法
    folders = [
        "/home/panxiang/coding/kweilx/ultralytics/_project/label_and_test_images/test_images/single_class/class_1/n_100_a_100_0/_anomaly",
        "/home/panxiang/coding/kweilx/ultralytics/_visual/heatmap/yolov8/data0", 
        "/home/panxiang/coding/kweilx/ultralytics/_visual/heatmap/new/data0"
    ]
    
    labels = ["source", "no-attention", "attention"]
    
    # 处理所有图像
    success, error = merge_all_images(
        source_list=folders,
        name_list=labels,
        resize=200,
        gap=15,
        output_dir="/home/panxiang/coding/kweilx/ultralytics/_visual/heatmap/results"
    )