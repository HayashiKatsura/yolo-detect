from PIL import Image, ImageDraw


def draw_rectangle(img, x, y, w, h, color=(0, 255, 0), thickness=2, savePath=None):
    # 确保颜色是RGB格
    image = img if isinstance(img, Image.Image) else Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    # Pillow中矩形的坐标需要两个点：左上角和右下角
    draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=thickness)
    image.save(savePath) if savePath else None
    return image


if __name__ == '__main__':
    PIC_PATH = r"D:\Coding\ultralytics\A_project\CustomTrain\datasets\images\train\burn_1.JPG"
    image = Image.open(PIC_PATH)
    image = draw_rectangle(image, 10, 10, 100, 100)
    image.show()
