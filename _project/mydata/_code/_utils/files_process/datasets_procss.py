import os
import shutil
import random


def is_txt_empty(txt_path):
    """
    判断标签文件是否为空
    Args:
        txt_path (_type_): _description_
    """
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                # 删掉这个文件
                os.remove(txt_path)
                print(f'Delete {txt_path}')
                return True
    except:
        return False
            

def makepair_datasets(source_folder):
    """
    删除不成对的图片或标签文件
    Args:
        source_folder (_type_): _description_
    """
    # 遍历标签
    for files in os.listdir(source_folder): 
        if str(files).lower().endswith('.txt'):
            # 先删除无效标签
            if not is_txt_empty(os.path.join(source_folder, files)):
                if not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(files)[0]}.png')) \
                and not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(files)[0]}.jpg')) \
                and not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(files)[0]}.jpeg')):
                    os.remove(os.path.join(source_folder, files))
                    print(f'Delete {os.path.join(source_folder, files)}')
    
    # 遍历图像
    for files in os.listdir(source_folder):
        if str(files).lower().endswith(('.jpg', '.png', '.jpeg')):
            if not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(files)[0]}.txt')):
                os.remove(os.path.join(source_folder, files))
                print(f'Delete {os.path.join(source_folder, files)}')

 
 
def devide_train_val_datasets(save_folder, source_folder, rate=0.3):
    """
    按比例划分数据集，并复制到指定文件夹
    Args:
        save_folder (_type_): _description_
        source_folder (_type_): _description_
        rate (float, optional): _description_. Defaults to 0.3.
    """
    file_separate = []
    images_folder = os.path.join(save_folder, 'images')  # 设置为你的图片目标文件夹路径
    labels_folder =  os.path.join(save_folder, 'labels') # 设置为你的标签目标文件夹路径

    # 确保目标文件夹存在
    for folder in [images_folder, labels_folder]:
        os.makedirs(os.path.join(folder, 'train'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'val'), exist_ok=True)

    # 收集符合条件的图片和标签文件对
    for files in os.listdir(source_folder):
        this_file_name = str(files).lower()
        if (this_file_name.endswith('.jpg') or this_file_name.endswith('.png')) and not this_file_name.endswith('_mask.png'):
            labels_file = os.path.join(source_folder, files.split('.')[0] + '.txt')
            if os.path.exists(labels_file):
                image_file = os.path.join(source_folder, files)
                file_separate.append({
                    'image_file': image_file,
                    'labels_file': labels_file
                })

    # 打乱文件顺序以实现随机分配
    random.shuffle(file_separate)

    # 计算训练集和验证集的分割点
    val_size = int(len(file_separate) * rate)
    train_size = len(file_separate) - val_size

    # 分配文件到训练集和验证集
    for i, file_pair in enumerate(file_separate):
        # 获取文件名（不包括路径）
        image_filename = os.path.basename(file_pair['image_file'])
        label_filename = os.path.basename(file_pair['labels_file'])
        
        # 确定目标子文件夹（训练集或验证集）
        target_subfolder = 'val' if i < val_size else 'train'
        
        # 复制图片文件
        if os.path.exists(os.path.join(images_folder, target_subfolder, image_filename)):
            # 先把该文件重命名
            image_filename = f"{str(image_filename).split('.')[0]}_copy.{str(image_filename).split('.')[1]}"
            label_filename = f"{str(label_filename).split('.')[0]}_copy.{str(label_filename).split('.')[1]}"
        
        target_image_path = os.path.join(images_folder, target_subfolder, image_filename)
        shutil.copy2(file_pair['image_file'], target_image_path)
        
        # 复制标签文件
        target_label_path = os.path.join(labels_folder, target_subfolder, label_filename)
        shutil.copy2(file_pair['labels_file'], target_label_path)

    # 打印统计信息
    print(f"总共处理了 {len(file_separate)} 对图片和标签文件")
    print(f"训练集: {train_size} 对文件 ({100 - rate * 100:.0f}%)")
    print(f"验证集: {val_size} 对文件 ({rate * 100:.0f}%)")
    

def batch_devide_train_val_datasets(save_folder_list, source_folder_list, rate=0.3):
    """
    批量处理数据集，并复制到指定文件夹,需要一一对应
    Args:
        save_folder_list (_type_): _description_
        source_folder_list (_type_): _description_
        rate (float, optional): _description_. Defaults to 0.3.
    """
    for save_fl, source_fl in zip(save_folder_list, source_folder_list):
        devide_train_val_datasets(save_fl, source_fl, rate=rate)




def remove_invalid_files(source_folder):
    for file in os.listdir(source_folder):
        if file.endswith('.txt'):
            with open(os.path.join(source_folder, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) == 0:
                    os.remove(os.path.join(source_folder, file))
                    if os.path.exists(os.path.join(source_folder, f'{os.path.splitext(file)[0]}.png')):
                        os.remove(os.path.join(source_folder, f'{os.path.splitext(file)[0]}.png'))
    
    for file in os.listdir(source_folder):
        if file.endswith('.png'):
            if not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(file)[0]}.txt')):
                os.remove(os.path.join(source_folder, file))
    
    for file in os.listdir(source_folder):
        if file.endswith('.txt'):
            if not os.path.exists(os.path.join(source_folder, f'{os.path.splitext(file)[0]}.png')):
                os.remove(os.path.join(source_folder, file))
    


def devide_train_val_folders(source_dir, train_dir, val_dir, train_ratio=0.7):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取源目录下所有.png文件和.txt文件
    png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    
    # 获取每个.png文件对应的.txt文件
    file_pairs = []
    for png in png_files:
        txt = os.path.splitext(png)[0] + '.txt'
        if txt in os.listdir(source_dir):  # 确保对应的.txt文件存在
            file_pairs.append((png, txt))
    
    # 打乱文件对的顺序
    random.shuffle(file_pairs)

    # 计算训练集和验证集的数量
    train_size = int(len(file_pairs) * train_ratio)

    # 将文件对移动到对应的文件夹
    for i, (png, txt) in enumerate(file_pairs):
        if i < train_size:
            # 移动到train文件夹
            shutil.move(os.path.join(source_dir, png), os.path.join(train_dir, png))
            shutil.move(os.path.join(source_dir, txt), os.path.join(train_dir, txt))
        else:
            # 移动到val文件夹
            shutil.move(os.path.join(source_dir, png), os.path.join(val_dir, png))
            shutil.move(os.path.join(source_dir, txt), os.path.join(val_dir, txt))

    print(f"Files have been split into {train_size} for training and {len(file_pairs) - train_size} for validation.")

   

if __name__ == '__main__':
    source_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies'
    source_list = ['0','1','2','3','4','5']
    save_list = ["black","damage","ink","resuide","pi","circle"]
    per_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/new_source'
    for img_label in ['images', 'labels']:
        this_folder = os.path.join(per_folder, img_label,'val')
        for anomalies_type in save_list:
            target_save_folder = os.path.join(source_folder, anomalies_type,img_label,'val')
            shutil.copytree(this_folder, target_save_folder)

    
    
   