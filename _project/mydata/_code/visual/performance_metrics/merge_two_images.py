import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def merge_images_horizontally(images_list, names_list=None, gap=10, save_path=None):
    """
    横向合并多张图像，并在每张图像下方添加名称
    
    参数:
        images_list (list): 图像列表，可以是PIL.Image对象或图像文件路径
        names_list (list|None): 图像名称列表，如果不提供则使用文件名
        gap (int): 两张图像之间的间距，默认为10像素
        save_path (str): 保存合并后图像的路径
        
    返回:
        PIL.Image: 合并后的图像对象
    """
    # 处理图像列表，确保所有元素都是PIL.Image对象
    processed_images = []
    for i, img in enumerate(images_list):
        if isinstance(img, str):
            # 如果是文件路径，打开图像
            processed_images.append(Image.open(img))
        else:
            # 假设已经是PIL.Image对象
            processed_images.append(img)
    
    # 处理名称列表
    if names_list is None:
        names_list = []
        for i, img in enumerate(images_list):
            if isinstance(img, str):
                # 如果图像是通过文件路径提供的，使用文件名作为默认名称
                names_list.append(os.path.basename(img))
            else:
                # 否则使用索引作为名称
                names_list.append(f"Image {i+1}")
    
    # 确保名称列表长度与图像列表相同
    if len(names_list) < len(processed_images):
        for i in range(len(names_list), len(processed_images)):
            names_list.append(f"Image {i+1}")
    
    # 设置字体和文本高度
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/PingFang-SC-Regular.ttf", 20)
    except IOError:
        # 如果无法加载特定字体，使用默认字体
        font = ImageFont.load_default()
        font.size = 60
    
    text_height = 60  # 文字区域高度
    
    # 计算合并后图像的尺寸
    max_height = max(img.height for img in processed_images)
    total_width = sum(img.width for img in processed_images) + gap * (len(processed_images) - 1)
    
    # 创建新图像
    merged_image = Image.new('RGB', (total_width, max_height + text_height), (255, 255, 255))
    draw = ImageDraw.Draw(merged_image)
    
    # 粘贴图像并添加名称
    x_offset = 0
    for i, (img, name) in enumerate(zip(processed_images, names_list)):
        # 在上半部分粘贴图像，垂直居中
        y_offset = (max_height - img.height) // 2
        merged_image.paste(img, (x_offset, y_offset))
        
        # 在图像下方添加名称
        # 使用新的方法计算文本尺寸
        bbox = font.getbbox(name)
        text_width = bbox[2] - bbox[0]
        
        text_x = x_offset + (img.width - text_width) // 2  # 文本水平居中
        text_y = max_height + 5  # 文本位置在图像下方
        draw.text((text_x, text_y), name, fill=(0, 0, 0), font=font)
        
        # 更新下一张图像的位置
        x_offset += img.width + gap
    
    # 保存合并后的图像
    if save_path:
        merged_image.save(save_path)
    
    return merged_image

if __name__ == '__main__':
    image_paths = ['/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/black/68A-01-Bin-065.png', 
                   '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/black/v12heat.png',
                   '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/black/chipsheat.png',
                   ]
    custom_names = ['Pollution', 'Circuit_damage','Circle','Pi_overexposure','Black']
    result = merge_images_horizontally(image_paths, 
                                       None, 
                                       gap=20, 
                                       save_path='/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/cases/black/mergeheat.png')

