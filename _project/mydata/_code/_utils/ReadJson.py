import json
import os
def read_json_file(filename):

    # 读取JSON文件（假设文件路径为'your_file.json'）
    with open(filename, 'r') as file:
        data = json.load(file)

    # 提取points的部分
    points_list = []  # 用来存储所有的points数据
    for shape in data.get('shapes', []):  # 遍历shapes列表
        points = shape.get('points', [])  # 获取每个shape的points字段
        if points:  # 如果points不为空
            points_list.append(points)  # 将points添加到列表中

    # 提取图像数据（图像可以是base64编码或文件路径）
    image_data = data.get('imageData', None)  # 获取base64编码的图像数据
    return {
        'points_list': points_list,
        'image_data': image_data
    }
    
def read_json_folder(foldername):
    results = []
    for files in os.listdir(foldername):
        try:
            if str(files).endswith('.json'):
                results.append(read_json_file(os.path.join(foldername,files)))
        except:
            continue
    print(f'json数量,{len(results)}')
    return results


if __name__ == '__main__':
    json_path = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/yolo_data/images_labels/ink_2.json'
    read_json_file(json_path)