import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

"""
只需要掩码，create_mask_from_points
需要黑白二值图，extract_region_from_image
需要保留原色彩且背景为白色时，extract_colored_region
"""

def decode_image(image_data):
    # 解码base64图像数据
    img_data = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_data))
    img = np.array(img)  # 转换为NumPy数组
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 转换为灰度图

def create_mask_from_points(points_list):
    """
    根据输入的点集合创建二值掩码（mask），返回掩码及其位置信息。
    """
    all_masks = []
    
    # 对每一个区域的点进行处理
    for points in points_list:
        points = np.array(points, dtype=np.int32)
        
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(points)
        
        # 创建一个全黑的图像
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 将points区域填充为白色
        points_shifted = points - [x, y]  # 将点坐标调整到外接矩形区域
        cv2.fillPoly(mask, [points_shifted], 255)
        
        # 将该区域的掩码添加到所有掩码列表中
        all_masks.append((mask, x, y, w, h))
    
    return all_masks

def extract_region_from_image(image_data, points_list):
    """
    利用 create_mask_from_points 创建的掩码，
    从原图中提取区域，并创建一个只有选定区域可见（白色）、
    其余部分为黑色背景的图像 -> mask_gt
    """
    img = decode_image(image_data)
    
    # 创建所有掩码图像
    all_masks = create_mask_from_points(points_list)
    
    # 创建一个全黑的图像，准备将所有区域合并
    extracted_image = np.zeros_like(img)
    
    # 对每个掩码进行合并
    for mask, x, y, w, h in all_masks:
        roi = np.zeros_like(img)  # 创建一个全黑的图像
        roi[y:y+h, x:x+w] = mask  # 将掩码图像的白色区域放入外接矩形区域
        extracted_image[y:y+h, x:x+w] = roi[y:y+h, x:x+w]
    
    # 返回合并后的图像，白色区域为提取的区域，背景为黑色
    return extracted_image


def extract_colored_region(image_data, points_list):
    """
    保持彩色信息抠图
    """
    # 解码图像（保留彩色信息）
    img_data = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_data))
    img = np.array(img)  # 转换为NumPy数组（保留彩色）
    
    # 创建一个白色背景的图像（与原图大小相同）
    white_bg = np.ones_like(img) * 255
    
    # 创建掩码
    all_masks = create_mask_from_points(points_list)
    
    for mask, x, y, w, h in all_masks:
        """
        对每个掩码区域进行处理
        x 和 y：掩码区域左上角在原图中的坐标位置
        w 和 h：掩码区域的宽度和高度
        """
        # 将掩码扩展为3通道（如果图像是彩色的）
        if len(img.shape) == 3:
            mask_3d = np.stack([mask] * img.shape[2], axis=2)
        else:
            mask_3d = mask
            
        # 对掩码区域应用原图像内容
        # 先获取区域
        region = img[y:y+h, x:x+w]
        
        # 如果维度不匹配，调整mask_3d维度
        if region.shape != mask_3d.shape and len(img.shape) == 3:
            mask_3d = np.stack([mask] * 3, axis=2)
        
        # 使用掩码将区域复制到白色背景
        region_with_mask = np.where(mask_3d > 0, region, 255)
        white_bg[y:y+h, x:x+w] = region_with_mask
    
    return white_bg


if __name__ == '__main__':
    from ReadJson import read_json_file
    json_path = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_data/multi_class/scratches_12.json'
    ct = read_json_file(json_path)
    points_list = ct.get('points_list')
    image_data =  ct.get('image_data')

    # # 提取区域
    # extracted_image = extract_region_from_image(image_data, points_list)
    # # 如果需要保存图像
    # cv2.imwrite('extracted_region.png', extracted_image)
    # white_image = extract_colored_region(image_data, points_list)
    # cv2.imwrite('extracted_region.png', white_image)
    # 先抠出彩色图像
    from ConcatImageAndAnomaly import concat_images
    extracted = extract_colored_region(image_data, points_list)

    normally_image = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_images/single_class/_normally/000.png'
    # 先抠出彩色图像

    # 叠加到新背景上（默认为背景图大小的0.2倍）
    result1,result1_mask = concat_images(extracted, normally_image,scale_factor=1)
    # cv2.imwrite('result1.png', result1)
    # 保存结果图像
    cv2.imwrite("result1.jpg", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))

    # 保存掩码图像
    cv2.imwrite("mask1.jpg", result1_mask)

    # 叠加到新背景上（50%透明度，0.3倍大小）
    result2,result2_mask = concat_images(extracted, normally_image, transparency=0.5, scale_factor=0.3)
    # cv2.imwrite('result2.png', result2)
    cv2.imwrite("result2.jpg", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))
    cv2.imwrite("mask2.jpg", result2_mask)