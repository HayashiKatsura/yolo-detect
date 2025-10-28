import os

from PIL import Image

from utils.DrawRectangle import draw_rectangle
from utils.GetPositionsFromYoloTxt import get_positions_from_yolo_data
from utils.PutTextOnPics import ShowTextOnPics
from utils.ReadTxt import read_txt


def restore_annotation_data(annotation_data_path=r"D:\Coding\项目资料\yolo_data"
                            , save_path=r"D:\Coding\项目资料\yolo_restore"):
    """
    绘制标注数据
    """
    for label_file in os.listdir(annotation_data_path):
        try:
            if label_file.endswith(".txt"):
                label_file_name = label_file.split(".")[0]
                label_file_path = os.path.join(annotation_data_path, label_file)
                image_file_1 = os.path.join(annotation_data_path, label_file_name + ".jpg")
                image_file_2 = os.path.join(annotation_data_path, label_file_name + ".png")
                image_file_3 = os.path.join(annotation_data_path, label_file_name + ".jpeg")
                image_file = image_file_1 if os.path.exists(image_file_1) else image_file_2 if os.path.exists(
                    image_file_2) else image_file_3
                text_data = read_txt(label_file_path)
                image = Image.open(image_file)
                width, height = image.size
                real_data = get_positions_from_yolo_data(text_data, height, width)
                for item in real_data:
                    label, [x1, y1], [x2, y2] = item
                    image = draw_rectangle(image, x1, y1, x2 - x1, y2 - y1, color=(0, 255, 0), thickness=2)
                    ShowTextOnPics().showTextOnPics(
                        imagePath=image,
                        positions=(x1, y1 - height // 20),
                        text=label,
                        fontColor=(0, 255, 0),
                        fontSize=height // 20,
                        savePath=os.path.join(save_path, label_file_name + ".jpg"))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    restore_annotation_data()
