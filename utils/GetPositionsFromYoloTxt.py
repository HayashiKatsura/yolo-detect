from utils.AmomalyClasses import getAnomalyClasses


def get_positions_from_yolo_data(yolo_data,img_height,img_width):
    """
    从yolo的txt文件中获取坐标信息
    """
    annotation_data = []
    for item in yolo_data:
        item = item.strip().split(' ')
        label = getAnomalyClasses().get(str(item[0]))
        x_center = float(item[1])
        y_center = float(item[2])
        width = float(item[3])
        height = float(item[4])
        # x1 = int((x_center - width / 2) * img_width)
        # y1 = int((y_center - height / 2) * img_height)
        # x2 = int((x_center + width / 2) * img_width)
        # y2 = int((y_center + height / 2) * img_height)
        # 计算中心点的实际像素坐标
        x_center = x_center * img_width
        y_center = y_center * img_height
        # 计算矩形框宽度和高度的实际像素大小
        box_width = width * img_width
        box_height = height * img_height
        # 计算左上角和右下角的像素坐标
        x_min = x_center - box_width / 2
        y_min = y_center - box_height / 2
        x_max = x_center + box_width / 2
        y_max = y_center + box_height / 2
        annotation_data.append([label,[x_min, y_min],[x_max, y_max]])
    print(annotation_data)
    return annotation_data



if __name__ == '__main__':
    yolo_data = ['3 0.071774 0.483871 0.140323 0.158065\n', '3 0.566129 0.483065 0.177419 0.156452\n', '3 0.565323 0.841129 0.220968 0.182258\n']
    get_positions_from_yolo_data(yolo_data,620,620)
