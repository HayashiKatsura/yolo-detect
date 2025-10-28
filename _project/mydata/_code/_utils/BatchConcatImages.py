"""
批量合成异常样本
"""
import cv2
from ReadJson import read_json_folder,read_json_file
from ConcatImageAndAnomaly import concat_images_with_positions
from ExtractAnomalyFromImages import extract_colored_region
import os
import random

def batch_concat_images(normal_images:str,json_folder:str,output_folder:str)->None:
    iter = 50
    # 114
    json_datas = read_json_folder(json_folder)
    for _iter in range(iter):
        for _files in os.listdir(normal_images):
            if str(_files).endswith('.png'):
                base_file_name = str(_files).split('.')[0]
                base_file_name = str(_iter) + base_file_name
                # 按顺序选中文件
                this_images = os.path.join(normal_images,_files)
                # 产生一个随机数
                idx = random.randint(0,len(json_datas)-1)
                points_list = json_datas[idx].get('points_list')
                image_data =  json_datas[idx].get('image_data')
                # 提取的异常元素
                extracted_anomaly = extract_colored_region(image_data, points_list)
                # 叠加到新背景上（默认为背景图大小的0.2倍）
                result,result_mask,yolo_label = concat_images_with_positions(extracted_anomaly, this_images,scale_factor=1)

                cv2.imwrite(os.path.join(output_folder,base_file_name+'.png'), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_folder,base_file_name+'_mask.png'), result_mask)
                with open(os.path.join(output_folder,base_file_name+'.txt'),'w') as f:
                    for sublist in yolo_label:
                        # 在每一行前面添加"0"，然后将子列表中的每个元素转为字符串并用空格分隔
                        line = "0 " + " ".join(map(str, sublist))
                        f.write(line + '\n')

def concat_image(normal_image:str,json_file:str,output_folder:str):
    json_data = read_json_file(json_file)
    points_list = json_data.get('points_list')
    image_data =  json_data.get('image_data')
    extracted_anomaly = extract_colored_region(image_data, points_list)
    cv2.imwrite('extracted_anomaly.png',cv2.cvtColor(extracted_anomaly, cv2.COLOR_RGB2BGR))
    result,result_mask,yolo_label = concat_images_with_positions(extracted_anomaly, normal_image,scale_factor=1)
    print()


if __name__ == '__main__':
    normal_images = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_images/single_class/_normally'
    json_folder = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_data/multi_class'
    output_folder = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_synthetic_images/_multi_class_much_more_cases'
    batch_concat_images(normal_images,json_folder,output_folder)
    
    # normal_image = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_images/single_class/_normally/000.png'
    # json_file = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_data/multi_class/scratches_12.json'
    # concat_image(normal_image,json_file,output_folder=None)  