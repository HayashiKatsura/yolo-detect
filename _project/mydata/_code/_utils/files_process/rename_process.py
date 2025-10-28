import os

def batch_rename_with_prefix(folder_path:str,extension:str|tuple,prefix:str):
    """
    批量重命名某些类型的文件，并添加前缀
    Args:
        folder_path (_type_): _description_
        extension (_type_): _description_
        prefix (_type_): _description_
    """

    extension = (extension) if isinstance(extension,str) else extension
    cnt = 0
    for filename in os.listdir(folder_path):
        if str(filename).lower().endswith(extension):
            file_name,file_extension = os.path.splitext(filename)
            # 获取文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, f"{prefix}{file_name}{file_extension}")
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已重命名: {filename} -> {new_file_path}")
            cnt += 1

    print(f"共{len(os.listdir(folder_path))}文件，重命名完成{cnt}个文件。")

def batch_rename_with_suffix(folder_path:str,extension:str|tuple,prefix:str):
    """
    批量重命名某些类型的文件，并添加后缀
    Args:
        folder_path (_type_): _description_
        extension (_type_): _description_
        prefix (_type_): _description_
    """

    extension = (extension) if isinstance(extension,str) else extension
    cnt = 0
    for filename in os.listdir(folder_path):
        if str(filename).lower().endswith(extension):
            file_name,file_extension = os.path.splitext(filename)
            # 获取文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, f"{file_name}{prefix}{file_extension}")
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已重命名: {filename} -> {new_file_path}")
            cnt += 1

    print(f"共{len(os.listdir(folder_path))}文件，重命名完成{cnt}个文件。")


def batch_rename_by_basename(folder_path,extension ,start_conut = 0, step = 1, fill = 3):
    """
    批量重命名文件，并按顺序编号
    Args:
        folder_path (_type_): _description_
        extension (_type_): _description_
        start_conut (_type_): _description_
        step (_type_): _description_
    """
    extension = (extension) if isinstance(extension,str) else extension
    cnt = 0
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if str(filename).lower().endswith(extension):
            file_name,file_extension = os.path.splitext(filename)
            new_file_name = f"{str(start_conut).zfill(fill)}{file_extension}"
            # 获取文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_file_name)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已重命名: {filename} -> {new_file_name}")
            start_conut += step
            cnt += 1

    print(f"共{len(os.listdir(folder_path))}文件，重命名完成{cnt}个文件。")
