import os
from random import sample

import numpy as np
from PIL import Image, ImageDraw

from utils.random_data import get_random_data, get_random_data_with_Mosaic
from utils.utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始数据集所在的路径
#   Out_VOCdevkit_path      输出数据集所在的路径
#-----------------------------------------------------------------------------------#
# Origin_VOCdevkit_path   = "VOCdevkit_Origin"
# Out_VOCdevkit_path      = "VOCdevkit"
Origin_VOCdevkit_path   = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies"
Out_VOCdevkit_path      = "/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_a_datasets/anomolies/4990source"


#-----------------------------------------------------------------------------------#
#   Out_Num                 利用mixup生成多少组图片
#   input_shape             生成的图片大小
#-----------------------------------------------------------------------------------#
Out_Num                 = 50
input_shape             = [1000, 1000]

if __name__ == "__main__":
    Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "4990source")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "5")
    
    Out_JPEGImages_path  = Out_VOCdevkit_path
    Out_Annotations_path = Out_VOCdevkit_path
    
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)
    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    # xml_names = os.listdir(Origin_Annotations_path)
    xml_names = [f for f in os.listdir(Origin_Annotations_path) if f.endswith('.txt')]
    
    
    def convert_to_yolo_format_and_save(box_data, save_path, image_width=input_shape[0], image_height=input_shape[1]):
        yolo_labels = []
        for box in box_data:
            xmin, ymin, xmax, ymax, cls_id = box
            
            # 计算 YOLO 格式的坐标
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # 生成对应的 YOLO 标签
            yolo_label = f"{int(cls_id)} {x_center} {y_center} {width} {height}"
            yolo_labels.append(yolo_label)
        
        # 将 YOLO 标签保存到文件
        with open(save_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')

        print(f"YOLO labels saved to {save_path}")
    
    #------------------------------#
    #   循环生成xml和jpg
    #------------------------------#
    for index in range(Out_Num):
        #------------------------------#
        #   获取4个图像与标签
        #------------------------------#
        sample_xmls     = sample(xml_names, 4)
        # unique_labels   = get_classes(sample_xmls, Origin_Annotations_path)

        annotation_line = []
        for xml in sample_xmls:
            # line = convert_annotation(os.path.join(Origin_JPEGImages_path, os.path.splitext(xml)[0] + '.jpg'), os.path.join(Origin_Annotations_path, xml), unique_labels)
            line = convert_annotation(os.path.join(Origin_JPEGImages_path, os.path.splitext(xml)[0] + '.png'), os.path.join(Origin_Annotations_path, xml))
            
            annotation_line.append(line)

        #------------------------------#
        #   合并mosaic
        #------------------------------#
        image_data, box_data = get_random_data_with_Mosaic(annotation_line, input_shape)
        
        img = Image.fromarray(image_data.astype(np.uint8))
        img.save(os.path.join(Out_JPEGImages_path, f'class_{os.path.basename(Origin_Annotations_path)}_mosaic_{index+100}.png'))
        convert_to_yolo_format_and_save(
            box_data,
            os.path.join(Out_Annotations_path, f'class_{os.path.basename(Origin_Annotations_path)}_mosaic_{index+100}.txt'),
        )
        # img.save(os.path.join(Out_JPEGImages_path, str(index) + '.jpg'))
        # write_xml(os.path.join(Out_Annotations_path, str(index) + '.xml'), os.path.join(Out_JPEGImages_path, str(index) + '.jpg'), \
        #             headstr, input_shape, box_data, unique_labels, tailstr)
