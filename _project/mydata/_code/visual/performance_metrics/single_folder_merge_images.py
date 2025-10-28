import os
import glob
from PIL import Image, ImageDraw, ImageFont

def combine_images(
    folder_path: str,
    images_per_row: int,
    image_spacing: int,
    image_width: int,
    file_names: list = None,
    save_path: str = "combined_image.jpg"
):
    """
    将文件夹中的所有图像合并到一个网格图像中。
    
    参数:
        folder_path (str): 包含图像的文件夹路径
        images_per_row (int): 每行放置的图像数量
        image_spacing (int): 图像之间的像素间距
        image_width (int): 每个图像的目标宽度（将按比例缩放）
        file_names (list, optional): 要在每个图像下方写入的名称列表。默认为None。
        save_path (str): 保存合并图像的路径
    """
    # 获取文件夹中的所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"在 {folder_path} 中未找到图像")
        return
    
    # 对图像文件进行排序以确保一致的顺序
    image_files.sort()
    
    # 随机选择九个元素
    import random
    image_files = random.sample(image_files, 6)
    
    
    # 如果未提供file_names，则使用文件名作为标签
    if file_names is None:
        file_names = [os.path.basename(f) for f in image_files]
    
    # 检查file_names是否与图像数量匹配
    if len(file_names) != len(image_files):
        print("错误：文件名数量与图像数量不匹配。")
        return
    
    # 打开第一个图像以获取原始尺寸
    sample_img = Image.open(image_files[0])
    orig_width, orig_height = sample_img.size
    
    # 根据纵横比计算新高度
    image_height = int(image_width * orig_height / orig_width)
    
    # 确定适当的重采样过滤器
    try:
        # 检查是否可用Resampling（Pillow >= 9.0.0）
        if hasattr(Image, 'Resampling'):
            resampling_filter = Image.Resampling.LANCZOS
        else:
            # 适用于较旧的Pillow版本
            resampling_filter = Image.LANCZOS
    except AttributeError:
        # 针对非常旧版本的回退选项
        resampling_filter = Image.ANTIALIAS
    
    # 加载并调整所有图像大小
    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        img = img.resize((image_width, image_height), resampling_filter)
        images.append(img)
    
    # 计算网格尺寸
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row  # 向上取整除法
    
    # 为图像下方的文本添加空间
    font_size = max(10, image_width // 40)  # 根据图像宽度调整字体大小
    text_height = font_size + 10  # 为文本添加额外的填充
    
    # 为合并图像创建空白画布
    combined_width = images_per_row * image_width + (images_per_row + 1) * image_spacing
    combined_height = num_rows * (image_height + text_height) + (num_rows + 1) * image_spacing
    
    combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)
    
    # 尝试加载字体，如果不可用则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # 在不同操作系统上尝试另一种常见字体
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
    # 在画布上放置每个图像
    for i, (img, name) in enumerate(zip(images, file_names)):
        row = i // images_per_row
        col = i % images_per_row
        
        # 计算位置
        x = image_spacing + col * (image_width + image_spacing)
        y = image_spacing + row * (image_height + text_height + image_spacing)
        
        # 粘贴图像
        combined_img.paste(img, (x, y))
        
        # 添加文本 - 处理不同的Pillow版本
        # 获取文本宽度以进行居中
        if hasattr(draw, 'textlength'):
            text_width = draw.textlength(name, font=font)
        else:
            # 适用于较旧Pillow版本的回退选项
            if hasattr(font, 'getsize'):
                text_width, _ = font.getsize(name)
            else:
                # 对于非常旧的版本或如果getsize不可用
                text_width = len(name) * (font_size // 2)  # 粗略估计
        
        text_x = x + (image_width - text_width) // 2  # 文本居中
        text_y = y + image_height + 5  # 位于图像下方
        draw.text((text_x, text_y), name, fill=(0, 0, 0), font=font)
    
    # 保存合并的图像
    combined_img.save(save_path)
    print(f"合并图像已保存至 {save_path}")
    
    # 使用示例
if __name__ == "__main__":
    folder_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolov8m/visualization_results/20250521/compare"  # 替换为实际路径
    images_per_row = 3                  # 每行图像数量
    image_spacing = 10                   # 图像间距（像素）
    image_width = 1000                    # 调整后的图像宽度
    save_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolov8m/visualization_results/20250521/compare/combined_output.jpg"    # 保存路径
    
    # 可选：自定义文件名（必须与图像数量匹配）
    file_names = ["case 1", "case 2", "case 3", "case 4", "case 5", "case 6"]
    
    combine_images(
        folder_path=folder_path,
        images_per_row=images_per_row,
        image_spacing=image_spacing,
        image_width=image_width,
        file_names=file_names,  # 取消注释以使用自定义名称
        save_path=save_path
    )