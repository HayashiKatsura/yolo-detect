import os
import shutil

def batch_copy_files(source_folder: str, target_folder: str, condition) -> None:
    """
    批量复制文件到目标文件夹，文件需要满足条件
    
    :param source_folder: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param condition: 一个函数，接受文件名和目标文件夹图像文件名作为输入，返回布尔值
    """
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 遍历源文件夹中的文件
    for filename in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, filename)
        
        # 只处理文件，忽略文件夹
        if os.path.isfile(source_file_path):
            # 检查文件是否满足条件
            if condition(filename, target_folder):
                # 目标文件路径
                target_file_path = os.path.join(target_folder, filename)
                
                # 执行复制操作
                shutil.copy(source_file_path, target_file_path)
                print(f"已复制文件: {filename}")

def txt_name_equals_image_name(filename: str, target_folder: str) -> bool:
    """
    文件复制条件：文件名以 .txt 结尾且与目标文件夹中的图像文件名一致
    
    :param filename: 源文件名
    :param target_folder: 目标文件夹路径
    :return: 是否符合条件
    """
    # 检查文件是否是以 .txt 结尾
    if filename.endswith('.txt'):
        # 获取目标文件夹中的图像文件
        image_files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f)) and f.lower().endswith(('jpg', 'png', 'jpeg'))]
        
        # 获取文件名和扩展名（不使用固定长度，使用 splitext）
        file_name, _ = os.path.splitext(filename)
        
        # 检查源文件名（不带扩展名）是否与图像文件名一致
        for image_file in image_files:
            image_name, _ = os.path.splitext(image_file)
            if file_name == image_name:
                return True
    return False



def copy_damage_files(source_folder: str, target_folder: str) -> None:
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            source_file_path = os.path.join(source_folder, filename)
            with open(source_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) > 0:
                    label = int(line[0])
                    if label == 1:
                        target_file_path = os.path.join(target_folder, filename)
                        shutil.copy(source_file_path, target_file_path)
                        shutil.copy(source_file_path.replace('txt', 'png'), 
                                    target_file_path.replace('txt', 'png'))
                    
import numpy as np

def bbox_iou(box1, box2):
    """计算两个YOLO框的IoU"""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return inter_area / (area1 + area2 - inter_area)

def non_max_suppression(boxes, class_ids, iou_threshold=0.5):
    """不考虑置信度，仅根据IoU进行位置去重"""
    kept_boxes = []
    kept_classes = []

    while boxes:
        curr_box = boxes.pop(0)
        curr_class = class_ids.pop(0)

        kept_boxes.append(curr_box)
        kept_classes.append(curr_class)

        new_boxes = []
        new_classes = []

        for i, box in enumerate(boxes):
            if class_ids[i] != curr_class or bbox_iou(curr_box, box) <= iou_threshold:
                new_boxes.append(box)
                new_classes.append(class_ids[i])
        boxes = new_boxes
        class_ids = new_classes

    return kept_boxes, kept_classes

def read_yolo_txt(file_path):
    boxes, class_ids = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            class_ids.append(class_id)
            boxes.append(coords)
    return boxes, class_ids

def write_yolo_txt(file_path, boxes, class_ids):
    with open(file_path, 'w') as f:
        for cls, box in zip(class_ids, boxes):
            f.write(f"{cls} {' '.join(map(str, box))}\n")

def process_yolo_labels(file_path, iou_threshold=0.5):
    boxes, class_ids = read_yolo_txt(file_path)
    final_boxes, final_classes = non_max_suppression(boxes, class_ids, iou_threshold)
    write_yolo_txt(file_path, final_boxes, final_classes)


def batch_process_yolo_labels(input_folder, iou_threshold=0.5):
    for file in os.listdir(input_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(input_folder, file)
            process_yolo_labels(file_path, iou_threshold)
    
def devide_dataset(source_folder,save_folder):
    for file in os.listdir(source_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(source_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) > 0:
                    label = int(line[0])
                    os.makedirs(os.path.join(save_folder,str(label)),exist_ok=True)
                    shutil.copy(file_path, os.path.join(save_folder,str(label),file))
                    print(f"已复制文件: {file}")

 
 
def move_random_files(source_dir, dest_dir, num_files=500):
    import random

    # 获取文件夹内所有.png和.txt文件
    png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    txt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]

    # 根据文件名匹配.png和.txt文件
    file_pairs = []
    for png in png_files:
        txt = png.replace('.png', '.txt')
        if txt in txt_files:
            file_pairs.append((png, txt))

    # 确保有足够的文件对
    if len(file_pairs) < num_files:
        print(f"文件对不足，只有 {len(file_pairs)} 对文件可用")
        num_files = len(file_pairs)

    # 随机选择指定数量的文件对
    selected_pairs = random.sample(file_pairs, num_files)

    # 移动文件到目标文件夹
    for png, txt in selected_pairs:
        # 移动.png文件
        shutil.move(os.path.join(source_dir, png), os.path.join(dest_dir, png))
        # 移动.txt文件
        shutil.move(os.path.join(source_dir, txt), os.path.join(dest_dir, txt))

    print(f"成功移动 {num_files} 对文件到 {dest_dir}")          
    
    
def batch_remove_files_by_suffix(source_folder,suffix):
    """
    批量删除文件夹中指定后缀的文件
    """
    cnt = 0
    for file in os.listdir(source_folder):
        if str(file).lower().split('.')[0].endswith(suffix):
            cnt+=1
            os.remove(os.path.join(source_folder,file))
            print(f"Remove {file} in {source_folder}")
    print(f"Total {cnt} files removed")
            


def batch_remove_lines_in_txt(source_folder, prefix):
    """
    删除文件中的指定前缀的行
    """
    cnt = 0
    for file in os.listdir(source_folder):
        if file.lower().endswith('.txt'):
            file_path = os.path.join(source_folder, file)
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 过滤掉以 prefix 开头的行
                new_lines = [line for line in lines if not line.startswith(prefix)]

                # 写回修改后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

                cnt += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Total {cnt} files were modified.")



def replace_line_starting_with_before_with_after(folder,before,after):
    """
    替换文件中的一行，如果以 before 开头，则替换为 after
    """
    
    cnt = 0
    for file in os.listdir(folder):
        if file.lower().endswith('class_5.txt'):
            file_path = os.path.join(folder, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 修改每一行：如果以 '5' 开头，替换第一列为 '4'
                new_lines = []
                for line in lines:
                    if line.startswith(before):  # 注意判断是否是以 "5" 后面带空格开头
                        new_line = after + line[len(before):]  # 替换第一个字符为 4
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)

                cnt += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Total {cnt} files were processed.")


def batch_rename_files(source_folder,extension):
    for files in os.listdir(source_folder):
        if str(files).lower().endswith(extension):
            if files.startswith('yolo11'):
                new_name = files.replace('yolo11','yolo12')
                os.rename(os.path.join(source_folder,files),os.path.join(source_folder,new_name))
                print(f"Rename {files} to {new_name}")
            

def get_different_images_1(folder_1,folder_2,val_folder,target_folder_1,target_folder_2):
    for files in os.listdir(val_folder):
        if str(files).lower().endswith('.png'):
            if files not in os.listdir(folder_1):
                shutil.copy(os.path.join(val_folder,files),os.path.join(target_folder_1,files))
                
    for files in os.listdir(val_folder):
        if str(files).lower().endswith('.png'):
            if files not in os.listdir(folder_2):
                shutil.copy(os.path.join(val_folder,files),os.path.join(target_folder_2,files))


def get_different_images_2(train_folder,val_folder,test_folder,target_folder):
    for files in os.listdir(train_folder):
        if str(files).lower().endswith('.png'):
            if files in os.listdir(test_folder):
                shutil.copy(os.path.join(train_folder,files),os.path.join(target_folder,files))
    for files in os.listdir(val_folder):
        if str(files).lower().endswith('.png'):
            if files in os.listdir(test_folder):
                shutil.copy(os.path.join(val_folder,files),os.path.join(target_folder,files))





if __name__ == '__main__':
    
    VAL= '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/images/val'
    PREDICT_12 = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/predict_0.83/predict_results_conf0.5'
    NOT_IN_12 = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/not_in_12'
    NOT_IN_CHIPS = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/not_in_chips'
    IN_CHIPS = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/in_chips'
    CHIPS_PREDICT = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/predict_chipsyolo/predict_results_conf0.25'
    VALIDATE_1 = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source_total/validate'
    VALIDATE_2 = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/validate'
    COMPARE_ORIGINAL = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/test_images/0.5/compareoriginal'
    
    # for files in os.listdir(VAL):
    #     if str(files).lower().endswith('.png'):
    #         if files not in os.listdir(PREDICT_12):
    #             shutil.copy(os.path.join(VAL,files),os.path.join(NOT_IN_12,files))
    #         if files not in os.listdir(CHIPS_PREDICT):
    #             shutil.copy(os.path.join(VAL,files),os.path.join(NOT_IN_CHIPS,files))
    
    
    for files in os.listdir(CHIPS_PREDICT):
        if str(files).lower().endswith('.txt'):
            with open(os.path.join(CHIPS_PREDICT,files), mode='r') as f:
                lines_chips = len(f.readlines())
            try:
                with open(os.path.join(PREDICT_12,files), mode='r') as f:
                    lines_12 = len(f.readlines())
            except:
                lines_12 = 0
                
            if lines_chips!= lines_12:
                try:
                    shutil.copy(os.path.join(CHIPS_PREDICT,f'{files.split(".")[0]}.png'),
                            os.path.join(COMPARE_ORIGINAL,f'{files.split(".")[0]} chipsyolo.png'))
                except:
                    pass
                
                try:
                    shutil.copy(os.path.join(PREDICT_12,f'{files.split(".")[0]}.png'),
                            os.path.join(COMPARE_ORIGINAL,f'{files.split(".")[0]} yolo12.png'))
                except:
                    pass
            else:
                pass
           
              
                

         
    

   
  

