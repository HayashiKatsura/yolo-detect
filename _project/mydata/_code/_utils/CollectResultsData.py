import csv
import os
import re
def save_csv(data: list, save_path: str) -> None:
    # 判断文件是否已存在
    file_exists = os.path.isfile(save_path)
    
    # 写入 CSV 文件
    with open(save_path, mode='a', newline='') as csvfile: 
        fieldnames = ['data','conf','iou0.3','iou0.5','iou0.7','miss','wrong','ap50','ap75','ap50-95','name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        # 写入数据
        writer.writerows(data)

    print(f"Results have been saved to {save_path}")

def collect_results_data(base_folder,
                        folder_1=[
                            'DATA_0',
                            'DATA_1',
                        ], folder_2= [
                            '_CONF_0.25',
                            '_CONF_0.5',
                            '_CONF_0.75',
                        ], folder_3= [
                            '0.25_VAL',
                            '0.5_VAL',
                            '0.75_VAL',
                        ]):
    # import copy
    # _base_folder = copy.deepcopy(base_folder)
    results = []
    file_name = os.path.basename(base_folder)
    data = None
    for f1 in folder_1:
        if str(f1).find('0') != -1:
            data = 'data0'
        else:
            data = 'data1'
        for f2,f3 in list(zip(folder_2,folder_3)):
            current_folder = os.path.join(base_folder,f'{f1}{f2}')
            # iou
            iou_miss_rate_values = []
            with open(os.path.join(current_folder,'iou_log.txt'), mode='r') as f:
                lines = f.readlines()
                
                miss_rate_pattern = re.compile(r'MISS_RATE:\s*([\d\.]+)')
                
                for line in lines:
                    match = miss_rate_pattern.search(line)
                    if match:
                        iou_miss_rate_values.append(float(match.group(1))) 
            # miss/wrong
            false_positive_rate_values = []
            miss_rate_values = []

            # Open and read the file
            with open(os.path.join(current_folder,'predict_result.txt'), 'r') as file:
                # Read the file line by line
                lines = file.readlines()
                
                # Regular expressions to find the relevant rates
                miss_rate_pattern = re.compile(r'漏检率:\s*([\d\.]+)')
                false_positive_pattern = re.compile(r'误检率:\s*([\d\.]+)')
                
                for line in lines:
                    # Search for the miss rate (漏检率)
                    miss_rate_match = miss_rate_pattern.search(line)
                    if miss_rate_match:
                        miss_rate_values.append(float(miss_rate_match.group(1)))
                    
                    # Search for the false positive rate (误检率)
                    false_positive_match = false_positive_pattern.search(line)
                    if false_positive_match:
                        false_positive_rate_values.append(float(false_positive_match.group(1)))
            
            # val
            with open(os.path.join(base_folder,'VAL',f3,'metrics.txt'), 'r') as file:
                # Read the file line by line
                lines = file.readlines()
                
                                # Initialize lists to store the mAP values
                map50_values = []
                map75_values = []
                map50_95_values = []
                maps_values = []

                # Open and read the file
                with open(os.path.join(base_folder,'VAL',f3,'metrics.txt'), 'r') as file:
                    # Read the file line by line
                    lines = file.readlines()
                    
                    # Regular expressions to find the relevant mAP values
                    map50_pattern = re.compile(r'mAP50:\s*([\d\.]+)')
                    map75_pattern = re.compile(r'mAP75:\s*([\d\.]+)')
                    map50_95_pattern = re.compile(r'mAP50-95:\s*([\d\.]+)')
                    maps_pattern = re.compile(r'mAPs:\s*\[\s*([\d\.]+)\s*\]')
                    
                    for line in lines:
                        # Search for each mAP value and append them to the respective lists
                        map50_match = map50_pattern.search(line)
                        if map50_match:
                            map50_values.append(float(map50_match.group(1)))
                        
                        map75_match = map75_pattern.search(line)
                        if map75_match:
                            map75_values.append(float(map75_match.group(1)))
                        
                        map50_95_match = map50_95_pattern.search(line)
                        if map50_95_match:
                            map50_95_values.append(float(map50_95_match.group(1)))
                        
                        maps_match = maps_pattern.search(line)
                        if maps_match:
                            maps_values.append(float(maps_match.group(1)))
            
            results.append({
                'data': data,
                'conf': f2.split('_CONF_')[1],
                'iou0.3': iou_miss_rate_values[0],
                'iou0.5': iou_miss_rate_values[1],
                'iou0.7': iou_miss_rate_values[2],
                'miss': miss_rate_values[0],
                'wrong': false_positive_rate_values[0],
                'ap50': map50_values[0],
                'ap75': map75_values[0],
                'ap50-95': map50_95_values[0],
                'name': file_name,
            })
    
    save_csv(results, '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolov8m/0429/results.csv') 
    return results

def batch_collect_results_data(outer_folder):
    for f in os.listdir(outer_folder):
        if os.path.isdir(os.path.join(outer_folder,f)):
            try:
                collect_results_data(os.path.join(outer_folder,f))
            except:
                continue
    
    
             
if __name__ == '__main__':
    base_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolov8m/0429'
    batch_collect_results_data(base_folder)
                
