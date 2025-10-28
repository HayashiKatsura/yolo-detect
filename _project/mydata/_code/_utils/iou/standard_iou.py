import os
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')
import csv
import os

"""
暂时默认六中类别
标准计算iou,返回所有类别的iou，以及最终的平均iou，并最终输出csv文件
使用了匈牙利算法来进行一对一匹配。
iou以真实标签为准
当ture_label不存在，而predict_label存在时，iou为-1
平均IoU（mIoU）是基于所有匹配成功的框计算的。
如果没有匹配成功，结果是0。
"""

from collections import defaultdict
import numpy as np

def compute_iou(box1, box2):
    # box = [cx, cy, w, h]
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def _calculate_class_iou(true_labels, predict_labels):
    # 按类别聚合
    true_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)

    for label in true_labels:
        cls, cx, cy, w, h = label
        true_by_class[int(cls)].append([cx, cy, w, h])

    for label in predict_labels:
        if len(label) != 5:
            label = label[:5]
        cls, cx, cy, w, h = label
        pred_by_class[int(cls)].append([cx, cy, w, h])

    iou_result = {}
    all_ious = []

    all_classes = set(true_by_class.keys()).union(pred_by_class.keys())

    for cls in all_classes:
        trues = true_by_class.get(cls, [])
        preds = pred_by_class.get(cls, [])
        matched = set()
        class_ious = []

        for t in trues:
            best_iou = 0
            best_idx = -1
            for idx, p in enumerate(preds):
                if idx in matched:
                    continue
                iou = compute_iou(t, p)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou > 0:
                matched.add(best_idx)
                class_ious.append(best_iou)

        # 类别IoU为平均IoU
        if class_ious:
            mean_class_iou = sum(class_ious) / len(class_ious)
        else:
            mean_class_iou = 0.0
        mean_class_iou = round(float(mean_class_iou), 2)
        iou_result[str(cls)] = mean_class_iou
        all_ious.append(mean_class_iou)

    iou_result['miou'] = round(float(sum(all_ious) / len(all_ious)),2) if all_ious else 0.0
    return iou_result


def calculate_class_iou(true_labels, predict_labels):
    # 按类别聚合
    true_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)

    for label in true_labels:
        cls, cx, cy, w, h = label
        true_by_class[int(cls)].append([cx, cy, w, h])

    for label in predict_labels:
        if len(label) != 5:
            label = label[:5]
        cls, cx, cy, w, h = label
        pred_by_class[int(cls)].append([cx, cy, w, h])

    iou_result = {}
    all_ious = []

    all_classes = set(true_by_class.keys()).union(pred_by_class.keys())

    for cls in all_classes:
        trues = true_by_class.get(cls, [])
        preds = pred_by_class.get(cls, [])

        # 🟡 特例：仅预测中出现该类（false positive 类别）
        if cls not in true_by_class:
            iou_result[str(cls)] = -1.0
            all_ious.append(0.0)
            continue

        matched = set()
        class_ious = []

        for t in trues:
            best_iou = 0
            best_idx = -1
            for idx, p in enumerate(preds):
                if idx in matched:
                    continue
                iou = compute_iou(t, p)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou > 0:
                matched.add(best_idx)
                class_ious.append(best_iou)

        # 类别IoU为平均IoU
        if class_ious:
            mean_class_iou = sum(class_ious) / len(class_ious)
        else:
            mean_class_iou = 0.0
            
        mean_class_iou = round(float(mean_class_iou), 2)
        iou_result[str(cls)] = mean_class_iou
        all_ious.append(mean_class_iou)

    # iou_result['miou'] = sum(all_ious) / len(all_ious) if all_ious else 0.0
    iou_result['miou'] = round(float(sum(all_ious) / len(all_ious)),2) if all_ious else 0.0
    return iou_result

def save_csv(data, save_path: str) -> None: 
    # 判断文件是否已存在
    file_exists = os.path.isfile(save_path)
    
    # 写入 CSV 文件
    with open(save_path, mode='a', newline='') as csvfile: 
        fieldnames = ['file_name', '0', '1', '2', '3', '4', '5', 'miou','method', 'data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 补全缺失字段
        complete_data = {key: data.get(key, 0.0) for key in fieldnames}
        
        # 写入数据
        writer.writerow(complete_data)

def main(predict_labels_path: str,
         true_labels_path: str | list[str],
         predict_labels_folder_names : str | list[str] = 'predict_results_conf0.8',
         ):
        """
        合并统计iou

        Args:
       
        """
        predict_labels_folder_names = [predict_labels_folder_names] if isinstance(predict_labels_folder_names, str) else predict_labels_folder_names
        for predict_item in predict_labels_path:
            for index, item in enumerate(['DATA_0_CONF_0.25_normal0']):
                
                true_item = true_labels_path[index]
                for true_label in sorted(os.listdir(true_item)):
                    if os.path.isfile(os.path.join(true_item,true_label)) and str(true_label).lower().endswith(('.txt')):
                        true_labels = [] # 真实标签
                        with open(os.path.join(true_item,true_label), 'r') as f:
                            _tlabels = f.readlines()
                            for _tlabel in _tlabels:
                                _tlabel = _tlabel.strip().split(' ')
                                true_labels.append([float(item) for item in _tlabel])
                                
                                predict_labels = [] # 预测标签
                                try:
                                    with open(os.path.join(predict_item,item,'predict_labels',true_label), 'r') as f:
                                        _plabels = f.readlines()
                                        for _plabel in _plabels:
                                            _plabel = _plabel.strip().split(' ')
                                            predict_labels.append([float(item) for item in _plabel])
                                except:
                                    continue
                            
                            iou_result = calculate_class_iou(true_labels, predict_labels)
                            method_name = 'v8' if str(os.path.basename(predict_item)).find('EfficientNetV1') == -1 else 'new'
                            method_name = str(os.path.basename(os.path.dirname(predict_item))) + '_' + method_name
                            iou_result.update({'file_name':f'{str(true_label).split(".")[0]}','method':method_name,'data':f'{item.split("0.25_")[1]}'})
                            save_csv(iou_result, os.path.join('/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202506151726_yolo12_Train301_V127_/DATA_0_CONF_0.25_normal0','iou_result.csv'))



def main(
        iou_data_list:dict|list[dict],
        true_labels_path: str | list[str],
        csv_save_path:str = None,
        method_name:str = 'baseline',
        data_type:str = 'default_data',
         ):
        """
        合并统计iou
        传入字典，或以字典为元素的列表
        字典中必须包含：
            predict_labels_path: 预测标签路径
            true_labels_path: 真实标签路径
            method_name: 方法名
            data_type: 数据类型
            csv_save_path: csv保存路径
            其中true_labels_path，csv_save_path，method_name，data_type可选，可以自行传入不同的， 或者是使用统一的数据
        Args:
            iou_data_list: 字典或以字典为元素的列表
            true_labels_path: 真实标签路径
            csv_save_path: csv保存路径
            method_name: 方法名
            data_type: 数据类型
        """
        iou_data_list = [iou_data_list] if isinstance(iou_data_list, dict) else iou_data_list

        
        for iou_data in iou_data_list:
            method_name = iou_data.get('method_name', method_name)
            data_type = iou_data.get('data_type', data_type)
            csv_save_path = iou_data.get('csv_save_path', csv_save_path)
            predict_labels_path = iou_data['predict_labels_path'] # 预测标签
            true_labels_path = iou_data.get('true_labels_path', true_labels_path) # 要么各传各的，要么就默认测试一个数据
            for true_label in sorted(os.listdir(true_labels_path)):
                if os.path.isfile(os.path.join(true_labels_path,true_label)) and str(true_label).lower().endswith(('.txt')):
                    true_labels = [] # 真实标签
                    with open(os.path.join(true_labels_path,true_label), 'r') as f:
                        _tlabels = f.readlines()
                        for _tlabel in _tlabels:
                            _tlabel = _tlabel.strip().split(' ')[:5]
                            true_labels.append([float(item) for item in _tlabel])
                            
                            predict_labels = [] # 预测标签
                            try:
                                with open(os.path.join(predict_labels_path,true_label), 'r') as f:
                                    _plabels = f.readlines()
                                    for _plabel in _plabels:
                                        _plabel = _plabel.strip().split(' ')[:5]
                                        predict_labels.append([float(item) for item in _plabel])
                            except:
                                continue
                        
                        iou_result = calculate_class_iou(true_labels, predict_labels)
                        iou_result.update({'file_name':f'{str(true_label).split(".")[0]}','method':method_name,'data':f'{data_type}'})
                        csv_save_path = os.path.dirname(predict_labels_path) if csv_save_path is None else csv_save_path
                        save_csv(iou_result, os.path.join(csv_save_path,'iou_result.csv'))
                    

                            

if __name__ == '__main__':
    iou_data_list =[
        {
            "predict_labels_path":"/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/predict/train/train_predict_results_conf0.25"
        }
    ]
    true_labels_path = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/labels/train"
    main(iou_data_list, true_labels_path)