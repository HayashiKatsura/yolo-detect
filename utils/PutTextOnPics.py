import os

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin


class ShowTextOnPics:
    """"
    只能测试本地文件
    不要用base64和图像矩阵
    """

    def __init__(self):
        self.fontPath = 'PingFang.ttc'

    def showTextOnPics(self,
                       imagePath,
                       positions,
                       text,
                       fontColor=(0, 255, 0),
                       fontSize=20,
                       isShow=False,
                       savePath=None):
        try:
            try:
                thisImage = Image.open(imagePath)
            except:
                thisImage = imagePath
            thisDraw = ImageDraw.Draw(thisImage)
            font = ImageFont.truetype(self.fontPath, fontSize)
            thisDraw.text((positions[0], positions[1]),
                          text,
                          font=font,
                          fill=fontColor)
            if isShow:
                thisImage.show() if isShow else None
            thisImage.save(savePath) if savePath else None
            return thisImage
            #
        except Exception as e:
            print(f"Error for :{e}")


if __name__ == '__main__':
    PIC_PATH = r"D:\Coding\ultralytics\A_project\CustomTrain\datasets\images\train\burn_1.JPG"
    test = ShowTextOnPics().showTextOnPics(PIC_PATH, (10, 10), "Hello World", isShow=False)
