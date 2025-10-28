import os
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')
import csv
import os
from typing import List
from scipy.optimize import linear_sum_assignment
import numpy as np

"""
只适用于Prof.Pan的医疗图像
使用了匈牙利算法来进行一对一匹配。
平均IoU（mIoU）是基于所有匹配成功的框计算的。
如果没有匹配成功，结果是0。
"""

# from .CalIou import calculate_iou, max_enclosing_rectangle

def check_labels(labels_folder:str,log_path:str):
    for files in os.listdir(labels_folder):
        if str(files).lower().endswith('.txt'):
            with open(os.path.join(labels_folder,files), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # 提取数据
                    data = line.strip().split()
                    if data[0] != '0':
                        with open(os.path.join(log_path, 'log.txt'), 'a') as f:
                            f.write(f"{files}\n")
                            break
                        
def yolo_to_bbox(cx, cy, w, h):
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return [xmin, ymin, xmax, ymax]


def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_mean_iou(file_name: str, gt_labels: List[List[float]], pred_labels: List[List[float]], csv_path: str):
    gt_boxes = [yolo_to_bbox(*gt[1:]) for gt in gt_labels]
    pred_boxes = [yolo_to_bbox(*pred[1:]) for pred in pred_labels]

    if not gt_boxes or not pred_boxes:
        miou = 0.0
    else:
        cost_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt in enumerate(gt_boxes):
            for j, pred in enumerate(pred_boxes):
                iou = compute_iou(gt, pred)
                cost_matrix[i][j] = -iou  # Negative for maximizing IoU

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        ious = []
        for i, j in zip(row_ind, col_ind):
            iou = -cost_matrix[i][j]  # Reverse sign
            if iou > 0:
                ious.append(iou)
        miou = np.mean(ious) if ious else 0.0

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Write or append to CSV
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['file_name', 'miou'])
        writer.writerow([file_name, round(miou, 6)])




def cal_iou_v2(true_labels,pred_labels,results_save,tag='train'):
    files_list = []
    for files in os.listdir(true_labels):
        files_list.append(files)
    files_list = sorted(files_list)
    for files in files_list:
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
                    _class = int(data[0])
                    _x = float(data[1])
                    _y = float(data[2])
                    _w = float(data[3])
                    _h = float(data[4])
                    true_cordinates.append([_class, _x, _y, _w, _h])
            
            try:
                with open(pred_label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # 提取数据
                        data = line.strip().split()
                        # 提取xmin、ymin、xmax、ymax
                        _class = int(data[0])
                        _x = float(data[1])
                        _y = float(data[2])
                        _w = float(data[3])
                        _h = float(data[4])
                        pred_cordinates.append([_class, _x, _y, _w, _h])
            except:
                pred_cordinates = []
            # 计算iou
            calculate_mean_iou(files.split('.')[0], true_cordinates, pred_cordinates, os.path.join(results_save, f'{tag}_iou.csv'))
            

                    


            



if __name__ == '__main__':
    # true_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/medical/202506111525/labels/train'
    # predict_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/medical/v8/DATA_0_CONF_0.25_train/predict_labels'
    true_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/multi_class/5_six_anomolies/labels/train'
    predict_labels = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202506151726_yolo12_Train301_V127_/DATA_0_CONF_0.25_normal0/predict_labels'
    log_path = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202506151726_yolo12_Train301_V127_/compare'
    results_save = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202506151726_yolo12_Train301_V127_/compare'
    # check_labels(labels_folder,log_path)
    cal_iou_v2(true_labels,predict_labels,results_save,'train')