import os

"""
iou_v1 算法有明显漏洞
"""     
     
     
def calculate_iou(box1, box2):
    """
    计算两个YOLO格式矩形框的交并比(IoU)
    
    参数:
    box1: YOLO格式的矩形坐标 [x_center, y_center, width, height]
    box2: YOLO格式的矩形坐标 [x_center, y_center, width, height]
    
    返回:
    float: 交并比值，范围[0, 1]
    """
    # 提取坐标
    x1_center, y1_center, w1, h1 = box1
    x2_center, y2_center, w2, h2 = box2
    
    # 计算矩形的边界
    x1_min = x1_center - w1/2
    y1_min = y1_center - h1/2
    x1_max = x1_center + w1/2
    y1_max = y1_center + h1/2
    
    x2_min = x2_center - w2/2
    y2_min = y2_center - h2/2
    x2_max = x2_center + w2/2
    y2_max = y2_center + h2/2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 检查是否有交集
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    # 计算交集面积
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算各自矩形的面积
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # 计算并集面积 = 两个矩形的总面积 - 交集面积
    union_area = box1_area + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / union_area
    
    return iou      
        
            

def max_enclosing_rectangle(_coordinates: list) -> list:
    """
    计算YOLO格式坐标列表中的最大外接矩形
    
    参数:
    _coordinates: 列表，每个元素是长度为4的列表 [x_center, y_center, width, height]，YOLO格式
    
    返回:
    list: 最大外接矩形的YOLO格式坐标 [x_center, y_center, width, height]
    """
    # 检查输入是否为空
    if not _coordinates or len(_coordinates) == 0:
        return [0, 0, 0, 0]
    
    # 初始化边界值
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    
    # 遍历所有坐标，找出最小和最大的x、y值
    for coord in _coordinates:
        # 确保每个坐标有4个元素
        if len(coord) != 4:
            raise ValueError("每个坐标必须有4个元素 [x_center, y_center, width, height]")
        
        x_center, y_center, width, height = coord
        
        # 计算矩形的左上角和右下角坐标
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        # 更新边界值
        min_x = min(min_x, x_min)
        max_x = max(max_x, x_max)
        min_y = min(min_y, y_min)
        max_y = max(max_y, y_max)
    
    # 计算外接矩形的宽度和高度
    enclosing_width = max_x - min_x
    enclosing_height = max_y - min_y
    
    # 计算外接矩形的中心点
    enclosing_x_center = min_x + enclosing_width / 2
    enclosing_y_center = min_y + enclosing_height / 2
    
    # 返回YOLO格式的坐标 [x_center, y_center, width, height]
    return [enclosing_x_center, enclosing_y_center, enclosing_width, enclosing_height]


def main(true_labels: str,pred_labels: str,results_save: str,_iou: float=0.3,missrate = False) -> None:
    """

    Args:
        true_labels (str): 真实标签路径
        pred_labels (str): 预测标签路径
        results_save (str): 日志保存路径
        _iou (float, optional): iou阈值. Defaults to 0.3.
    """
    # global_csv_result = []
    total_cnt = 0
    iou_cnt = 0
    miss_checked = []
    for files in os.listdir(true_labels):
        try:
            if str(files).lower().endswith('.txt'):
                true_label_path = os.path.join(true_labels, files)
                pred_label_path = os.path.join(pred_labels, files)
                # 逐行读取数据
                true_cordinates = []
                pred_cordinates = []
                with open(true_label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # 提取数据
                        data = line.strip().split()
                        # 提取xmin、ymin、xmax、ymax
                        _x = float(data[1])
                        _y = float(data[2])
                        _w = float(data[3])
                        _h = float(data[4])
                        true_cordinates.append([_x, _y, _w, _h])
                
                with open(pred_label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # 提取数据
                        data = line.strip().split()
                        # 提取xmin、ymin、xmax、ymax
                        _x = float(data[1])
                        _y = float(data[2])
                        _w = float(data[3])
                        _h = float(data[4])
                        pred_cordinates.append([_x, _y, _w, _h])
                
                # 计算两个最大外接矩形的IoU
                iou = calculate_iou(max_enclosing_rectangle(true_cordinates), 
                                    max_enclosing_rectangle(pred_cordinates))
                if iou<=_iou:
                    miss_checked.append(f"{str(files).split('.')[0]}.png")
                    iou_cnt += 1
        except:
            iou_cnt += 1
            miss_checked.append(f"{str(files).split('.')[0]}.png")
            continue
        finally:
            total_cnt += 1
    miss_rate = iou_cnt/total_cnt
    # global_csv_result.append({
        
    # })
    with open(os.path.join(results_save, 'iou_log.txt'), 'a') as f:
        f.write(f"IOU_THRESHOLD: {_iou}\n")
        f.write(f"TOTAL_COUNT: {total_cnt},MISS_COUNT: {iou_cnt},MISS_RATE: {miss_rate}\n")
        f.write("MISS_CHECHKED_LIST:\n")
        for item in list(sorted(miss_checked)):
            f.write(f"{item}\n")
    if missrate:
        return miss_checked,miss_rate
    return miss_checked



if __name__ == '__main__':
    true_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/label_and_test_images/test_images/single_class/class_1/n_100_a_100_0/labels'
    pred_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo11m/single_class/single_anomaly/train70_val30_202503071351/iou_0.5/predict_labels'
    results_save = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo11m/single_class/single_anomaly/train70_val30_202503071351/iou_0.5'
    for iou in [0.3,0.5,0.7]:
        main(true_labels,pred_labels,results_save,iou)

