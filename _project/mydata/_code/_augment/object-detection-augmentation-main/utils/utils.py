import copy
import os
import xml.etree.ElementTree as ET

def get_classes(sample_xmls, Origin_Annotations_path):
    unique_labels  = []
    for xml in sample_xmls:
        in_file = open(os.path.join(Origin_Annotations_path, xml), encoding='utf-8')
        tree    = ET.parse(in_file)
        root    = tree.getroot()
        
        for obj in root.iter('object'):
            cls     = obj.find('name').text
            if cls not in unique_labels:
                unique_labels.append(cls)
    return unique_labels

def _convert_annotation(jpg_path, xml_path, classes):
    in_file = open(xml_path, encoding='utf-8')
    tree    = ET.parse(in_file)
    root    = tree.getroot()
    
    line = copy.deepcopy(jpg_path)
    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None and hasattr(obj, "text"):
            difficult = obj.find('difficult').text
        if int(difficult)==1:
            continue
        
        cls     = obj.find('name').text
        cls_id = classes.index(cls)
        
        xmlbox  = obj.find('bndbox')
        b       = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        
        line += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    return line

def convert_annotation(jpg_path, txt_path):
    from PIL import Image
    # 打开图像，获取宽度和高度
    image = Image.open(jpg_path)
    image_width, image_height = image.size
    
    line = copy.deepcopy(jpg_path)
    with open(txt_path, 'r') as f:
        line_contents = f.readlines()
        for line_content in line_contents:
            current_line = line_content.strip().split()
            if len(current_line) >= 5 and current_line[0] == os.path.basename(os.path.dirname(txt_path)):
                cls_id = int(current_line[0])  # 类别 ID
                x_center = float(current_line[1])
                y_center = float(current_line[2])
                width = float(current_line[3])
                height = float(current_line[4])

                # 计算原始坐标
                xmin = int((x_center - width / 2) * image_width)
                ymin = int((y_center - height / 2) * image_height)
                xmax = int((x_center + width / 2) * image_width)
                ymax = int((y_center + height / 2) * image_height)
                
                # 校正坐标：确保 xmin <= xmax 和 ymin <= ymax
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                if ymin > ymax:
                    ymin, ymax = ymax, ymin
                
                # 将转换后的坐标添加到输出中
                line += " " + ",".join([str(xmin), str(ymin), str(xmax), str(ymax)]) + ',' + str(cls_id)
    
    return line
