from PIL import Image
import os
from typing import List

def merge_images_vertically(image_paths: List[str], spacing: int = 10, output_path: str = "merged_image.png"):
    """
    宽度必须一致
    将多张图片按上下顺序拼接成一张图片
    
    参数:
    image_paths: 图片文件路径列表
    spacing: 图片之间的间距，默认为10像素
    output_path: 输出文件路径
    """
    
    # 检查输入参数
    if not image_paths:
        raise ValueError("图片路径列表不能为空")
    
    # 检查所有文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    # 加载所有图片
    images = []
    try:
        for path in image_paths:
            img = Image.open(path)
            # 转换为RGBA模式以处理透明度
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            images.append(img)
    except Exception as e:
        raise ValueError(f"加载图片时出错: {e}")
    
    # 检查宽度是否一致
    width = images[0].width
    for i, img in enumerate(images):
        if img.width != width:
            raise ValueError(f"第{i+1}张图片的宽度({img.width})与第一张图片的宽度({width})不一致")
    
    # 计算总高度：所有图片高度之和 + 间距
    total_height = sum(img.height for img in images)
    if len(images) > 1:
        total_height += spacing * (len(images) - 1)
    
    # 创建新的空白图片
    merged_image = Image.new('RGBA', (width, total_height), (255, 255, 255, 0))
    
    # 按顺序粘贴图片
    current_y = 0
    for img in images:
        merged_image.paste(img, (0, current_y))
        current_y += img.height + spacing
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 如果输出格式不支持透明度，转换为RGB
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in ['.jpg', '.jpeg']:
        # 创建白色背景
        rgb_image = Image.new('RGB', merged_image.size, (255, 255, 255))
        rgb_image.paste(merged_image, mask=merged_image.split()[-1])
        rgb_image.save(output_path, quality=95)
    else:
        merged_image.save(output_path)
    
    print(f"图片拼接完成！")
    print(f"输出文件: {output_path}")
    print(f"尺寸: {width} x {total_height}")
    print(f"拼接了 {len(images)} 张图片")


# 使用示例
if __name__ == "__main__":
    # 示例用法1：基本使用
    image_files = [
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/black/h-1.jpg", 
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/circle/h-1.jpg", 
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/damage/h-1.jpg",
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/ink/h-1.jpg",
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/pi/h-1.jpg",
                "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/residue/h-1.jpg",
                   ]
    merge_images_vertically(image_files, spacing=10, output_path="/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/merged_output.png")
    
    # 示例用法2：自定义间距
    # merge_images_vertically(image_files, spacing=20, output_path="merged_with_spacing.png")
    
    # 示例用法3：无间距拼接
    # merge_images_vertically(image_files, spacing=0, output_path="merged_no_spacing.png")
    
    # 示例用法4：输出为JPEG格式
    # merge_images_vertically(image_files, spacing=10, output_path="merged_output.jpg")