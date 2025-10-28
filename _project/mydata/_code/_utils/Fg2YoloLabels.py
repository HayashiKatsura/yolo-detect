import cv2
import numpy as np
import os
from pathlib import Path

def detect_anomalies_and_save_labels(input_dir, output_dir, threshold=30, 
                                   min_area=100, max_area=10000, 
                                   morph_kernel_size=5):
    """
    改进版异常区域检测，解决区域分割问题
    
    参数新增:
        morph_kernel_size: 形态学操作核大小 (默认5)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for img_file in os.listdir(input_dir):
        print(f"\n正在处理文件: {img_file}")
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print("无法读取图片，跳过")
            continue
            
        h, w = img.shape[:2]
        if h != 512 or w != 512:
            print(f"非标准尺寸图片，跳过")
            continue
        
        # 获取背景颜色并计算差异
        bg_color = img[0, 0]
        diff = cv2.absdiff(img, bg_color)
        sum_diff = np.sum(diff, axis=2)
        
        # 二值化处理
        _, binary = cv2.threshold(sum_diff.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
        
        # 增强形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 孔洞填充处理
        contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        for cnt in contours:
            cv2.drawContours(filled, [cnt], 0, 255, -1)  # -1表示填充轮廓
        
        # 最终轮廓检测
        final_contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 处理检测结果
        yolo_lines = []
        for cnt in final_contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            area = box_w * box_h
            
            # if area < min_area or area > max_area:
            #     continue
                
            # YOLO格式转换
            x_center = (x + box_w/2) / 512
            y_center = (y + box_h/2) / 512
            width = box_w / 512
            height = box_h / 512
            
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 输出结果
        detected_num = len(yolo_lines)
        status = "正常" if detected_num ==0 else f"发现{detected_num}个异常"
        print(f"处理结果: {status}")
        
        # 保存标签
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))


if __name__ == "__main__":
    # 使用示例
    input_folder = "/home/panxiang/coding/kweilx/ultralytics/_project/s_anomaly/fg"
    output_folder = "/home/panxiang/coding/kweilx/ultralytics/_project/s_anomaly/txt"
    
    # 调用处理函数（可根据需要调整参数）
    detect_anomalies_and_save_labels(
        input_dir=input_folder,
        output_dir=output_folder,
        threshold=30,     # 根据实际情况调整
        min_area=100,     # 最小区域像素
        max_area=10000    # 最大区域像素
    )