def read_txt(file_path):
    """
    读取txt文件内容
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

if __name__ == '__main__':
    file_path = r"D:\Coding\项目资料\yolo_data\ink_3.txt"
    read_txt(file_path)