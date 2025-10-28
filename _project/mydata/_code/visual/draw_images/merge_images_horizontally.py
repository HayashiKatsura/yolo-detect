from PIL import Image, ImageDraw, ImageFont
import os

def merge_images_horizontally(file_list, normalize_size=200, spacing=20, name_list=None, save_path="merged_image.png"):
    """
    水平合并多个图像
    
    参数:
    file_list: 文件列表，例如['1.png', '2.png', '3.png']
    normalize_size: 归一化尺寸，默认200（将所有图像统一缩放到200*200）
    spacing: 间距，默认20（两张图像之间的间距）
    name_list: 名称列表，在每张子图下写上名称
    save_path: 保存路径
    """
    
    if not file_list:
        raise ValueError("文件列表不能为空")
    
    # 检查所有文件是否存在
    for file_path in file_list:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载和处理图像
    images = []
    for file_path in file_list:
        try:
            img = Image.open(file_path)
            # 转换为RGBA模式以支持透明度
            img = img.convert('RGBA')
            # 缩放到指定尺寸
            img = img.resize((normalize_size, normalize_size), Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            raise ValueError(f"无法处理图像 {file_path}: {str(e)}")
    
    # 检查名称列表是否有效
    show_names = name_list is not None and len(name_list) == len(file_list)
    
    # 计算文字区域高度
    text_height = 30 if show_names else 0
    text_margin = 10 if show_names else 0
    
    # 计算合并后的图像尺寸
    total_width = len(images) * normalize_size + (len(images) - 1) * spacing
    total_height = normalize_size + text_height + text_margin
    
    # 创建新的空白图像
    # merged_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))
    # 创建新的空白图像（RGB模式，白色背景）
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    
    # 粘贴图像
    current_x = 0
    for i, img in enumerate(images):
        # 粘贴图像
        merged_image.paste(img, (current_x, 0))
        
        # 如果需要显示名称，在图像下方添加文字
        if show_names:
            draw = ImageDraw.Draw(merged_image)
            
            # 尝试使用系统字体，如果失败则使用默认字体
            try:
                # 在不同系统上尝试不同的字体
                font_paths = [
                    "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/performance_metrics/PingFang-SC-Regular.ttf", 
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 16)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
                    
            except Exception:
                font = ImageFont.load_default()
            
            # 获取文字尺寸
            text = str(name_list[i])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            
            # 计算文字位置（居中）
            text_x = current_x + (normalize_size - text_width) // 2
            text_y = normalize_size + text_margin
            
            # 绘制文字
            draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)
        
        # 移动到下一个位置
        current_x += normalize_size + spacing
    
    # 保存图像
    try:
        # 如果保存为PNG，保持RGBA模式
        if save_path.lower().endswith('.png'):
            merged_image.save(save_path, 'PNG')
        else:
            # 其他格式转换为RGB
            rgb_image = Image.new('RGB', merged_image.size, (255, 255, 255))
            rgb_image.paste(merged_image, mask=merged_image.split()[-1])
            rgb_image.save(save_path)
        
        print(f"图像已成功保存到: {save_path}")
        
    except Exception as e:
        raise ValueError(f"保存图像失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # # 示例1：基本使用
    # file_list = ["1.png", "2.png", "3.png"]
    # merge_images_horizontally(file_list, save_path="merged_basic.png")
    
    # 示例2：带名称标签
    source_folder = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/merge"
    save_path = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/temp'
    # save_path = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/test_images'
    
    # for files in sorted(os.listdir(source_folder)):
    #     if str(files).lower().endswith('chipsyolo.png'):
    #         if os.path.exists(os.path.join(source_folder, str(files).replace('chipsyolo','yolo12'))):
    #             file_list =[
    #                 f"/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/images/val/{files.replace(' chipsyolo','')}",
    #                 # f"/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/merge/{str(files).replace('chipsyolo','yolo12')}",
    #                 f"/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/merge/{files}",
    #             ]
            
    file_list =  [
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/temp/sp0.001.png",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/temp/sp0.01.png",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/temp/sp0.1.png",
        "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_code/visual/draw_images/temp/sp0.05.png"
    ]
    name_list = [
                '(a) lamada=0.001',
                '(b) lamada=0.01',
                '(c) lamada=0.01',
                '(d) lamada=0.05'
                ]
    try:
        merge_images_horizontally(file_list, 
                                name_list=name_list, 
                                normalize_size=400,
                                save_path=os.path.join(save_path, f"merged_sp.png")
                                )
    except:
        print(f"无法合并图像: {file_list[0]}")
    
    # # 示例3：自定义参数
    # file_list = ["1.png", "2.png", "3.png"]
    # name_list = ["First", "Second", "Third"]
    # merge_images_horizontally(
    #     file_list=file_list,
    #     normalize_size=150,
    #     spacing=30,
    #     name_list=name_list,
    #     save_path="merged_custom.png"
    # )