import zipfile
import py7zr
import os

def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def extract_rar(file_path, extract_to):
    import rarfile  # 需要先安装: pip install rarfile
    with rarfile.RarFile(file_path) as rf:
        rf.extractall(extract_to)

def extract_7z(file_path, extract_to):
    # 方法1: 使用py7zr库
    try:
        import py7zr  # 需要先安装: pip install py7zr
        with py7zr.SevenZipFile(file_path, mode='r') as z:
            z.extractall(path=extract_to)
    except ImportError:
        # 方法2: 使用pyunpack (它依赖patool)
        try:
            from pyunpack import Archive  # 需要先安装: pip install pyunpack patool
            Archive(file_path).extractall(extract_to)
        except ImportError:
            # 方法3: 调用7z命令行工具
            import subprocess
            subprocess.run(['7z', 'x', file_path, f'-o{extract_to}'], check=True)