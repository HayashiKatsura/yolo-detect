import os
import shutil

def replace_line_starting_with_before_with_after(folder,before,after):
    """
    替换文件中的一行，如果以 before 开头，则替换为 after
    Args:
        folder (str): 文件夹路径
        extension (str or tuple): 文件后缀名，可以是字符串或元组
        before (str): 要替换的字符串
        after (str): 替换后的字符串
    """
    cnt = 0
    for file in os.listdir(folder):
        if file.lower().endswith('.txt'):
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
    

def remove_lines_in_txt(source_folder, prefix):
    """
    删除文件中的指定前缀的行
    Args:
        source_folder (str): 文件夹路径
        prefix (str): 要删除的行的前缀
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


def cal_types_count(source_folder, save_path):
    """
    统计文件夹中各类别的文件数量，并将结果写入save_path.txt文件
    """
    category_count = {}  # 用于存储每个类别出现的文件数量

    # 遍历文件夹中的所有文件
    for files in os.listdir(source_folder):
        if files.endswith('.txt'):  # 只处理txt文件
            file_path = os.path.join(source_folder, files)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 用一个集合存储该文件中出现的类别
            file_categories = set()
            for line in lines:
                line = line.strip().split()
                if len(line) > 0:  # 如果行不为空
                    label = int(line[0])  # 获取类别（yolo格式，第一个值是类别）
                    file_categories.add(label)  # 将该类别添加到集合中，避免重复

            # 对于该文件中所有的类别，增加统计
            for label in file_categories:
                if label in category_count:
                    category_count[label] += 1  # 如果已经有这个类别，文件数+1
                else:
                    category_count[label] = 1  # 否则新增该类别，并设置为1

    # 将结果写入save_path.txt文件
    with open(save_path, 'w', encoding='utf-8') as save_file:
        # 对字典按类别(label)进行排序
        for label, count in sorted(category_count.items()):
            save_file.write(f'类别 {label} 出现的文件数量: {count}\n')


def remove_last_column_in_txt(source_folder):
    # 遍历文件夹中的所有.txt文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(source_folder, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 处理每一行，删除最后一列
            updated_lines = []
            for line in lines:
                # 分割行的内容，删除最后一列
                columns = line.split()
                if len(columns) > 5:  # 确保该行至少有6列
                    updated_line = ' '.join(columns[:5])  # 保留前五列
                    updated_lines.append(updated_line + '\n')
                else:
                    updated_lines.append(line)  # 保持原行不变

            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)
            
            print(f"Processed {filename}")

import glob
def modify_first_column_to_zero(folder_path,replace_text = '1'):
    """
    修改指定文件夹中所有txt文件的每一行的第一列数字为 replace_text
    
    Args:
        folder_path (str): 包含txt文件的文件夹路径
    """
    # 获取文件夹中所有的txt文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    # 统计处理的文件数量
    total_files = len(txt_files)
    processed_files = 0
    modified_lines = 0
    
    print(f"找到 {total_files} 个txt文件需要处理")
    
    # 遍历处理每个txt文件
    for txt_file in txt_files:
        try:
            # 读取文件的所有行
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            # 处理每一行
            modified_file_lines = []
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if parts and parts[0]!= '0':  # 确保行不为空且有内容可分割
                        # TODO
                        parts[0] = replace_text  # 将第一列替换为0
                        modified_line = " ".join(parts)
                        modified_file_lines.append(modified_line)
                        modified_lines += 1
                    else:
                        modified_file_lines.append(line)  # 保留原始行
                else:
                    modified_file_lines.append(line)  # 保留空行
            
            # 写回文件
            with open(txt_file, 'w') as f:
                f.write("\n".join(modified_file_lines))
                # 如果最后一行原来有换行符，添加回去
                if lines and lines[-1].endswith("\n"):
                    f.write("\n")
            
            processed_files += 1
                
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {str(e)}")
    
    print(f"成功处理 {processed_files}/{total_files} 个文件")
    print(f"共修改 {modified_lines} 行数据，所有行的第一列已替换为{replace_text}")

def count_by_condition(source_folder, condition):
    source_files = os.listdir(source_folder)
    count = 0
    for file in source_files:
        if condition(file):
            count += 1
    print(f"找到 {count} 个文件满足条件")
    return count



if __name__ == '__main__':   
    source_folder = '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/6_Classes_With_Augment_Datasets/damage/labels/train'
    def _condition_1(file):
        if str(file).lower().endswith('.txt'):
            with open(os.path.join(source_folder, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) > 1:
                print(str(file).split('.')[0])
                return True
            
    def _condition_2(file):
        if str(file).lower().endswith('.txt'):
            with open(os.path.join(source_folder, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) == 1 and lines[0].strip().split()[0] == '1':
                return True
            
    def _condition_3(file):
        if str(file).lower().endswith('.txt'):
            file_path = os.path.join(source_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 如果没有行（空文件），根据需求决定是否返回 True 还是 False
            if not lines:
                return False

            # 所有非空行的第一个字符都必须是 '1'
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:  # 跳过空行
                    continue
                if stripped_line.split()[0] != '1':
                    return False
            print(str(file).split('.')[0])
            return True

        return False

    def _condition_4(file):
        if str(file).lower().endswith('.txt'):
            with open(os.path.join(source_folder, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if len(lines) <= 0:
                return True
            
    def _condition_5(file):
        if str(file).lower().endswith('.txt'):
            if str(file).find('mosaic')!=-1:
                return True
            
    def _condition_6(file):
        if str(file).lower().endswith('.txt'):
            if str(file).find('mixup')!=-1:
                return True
            
    def _condition_7(file):
        if str(file).lower().endswith('.txt'):
            if str(file).find('random')!=-1:
                return True
    
    def _condition_8(file):
        if str(file).lower().endswith('.txt'):
            file_path = os.path.join(source_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                stripped_line = line.strip()
                if stripped_line and stripped_line.split()[0] == '1':
                    print(str(file).split('.')[0])
                    return True

            return False

        return False
            
    # count_by_condition(source_folder,_condition_1)
    # count_by_condition(source_folder,_condition_2)
    # count_by_condition(source_folder,_condition_3)
    count_by_condition(source_folder,_condition_4)
    
    # count_by_condition(source_folder,_condition_5)
    # count_by_condition(source_folder,_condition_6)
    # count_by_condition(source_folder,_condition_7)
    # count_by_condition(source_folder,_condition_8)
    
    
    
    
    