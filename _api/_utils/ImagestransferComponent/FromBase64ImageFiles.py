import cv2

from _api._utils.ImagestransferComponent.TransferImageFilesMain import TransferImageFiles


class TransferBase64ImageFiles(TransferImageFiles):
    """
    接收Base64字符串
    """

    def __init__(self, imageFiles):
        super().__init__(imageFiles)
        self.imageFiles = imageFiles

    def toNdarray(self):
        import numpy as np
        # 转换PIL图像到numpy数组
        return np.array(self.toPILData())

    def toPILData(self):
        from PIL import Image
        return Image.open(self.toBytes())

    def toBytes(self):
        import base64
        from io import BytesIO
        # 解码Base64字符串到二进制数据
        imageData = base64.b64decode(self.imageFiles)
        # 使用BytesIO将这个二进制数据转换为文件对象
        return BytesIO(imageData)

    def toLocalFiles(self, savePath=None):
        if savePath:
            try:
                self.toPILData().save(savePath)
                print('sucess')
            except Exception as e:
                print(f"{e}")

    def toBase64(self):
        return self.imageFiles


if __name__ == '__main__':
    from _api._utils.base64code.base64encode import image_to_base64

    # PIC_PATH = "/Users/hayashi/Documents/HL-information/DocumentRocogntioByPython/pics/bankcard/bankcard1.jpg"
    # PIC_PATH = image_to_base64(PIC_PATH)


    transferBase64ImageFiles = TransferBase64ImageFiles(PIC_PATH)
    img = transferBase64ImageFiles.toNdarray()
    cv2.imshow("image", img)
    # print(type(transferBase64ImageFiles.toNdarray()))
    # print(type(transferBase64ImageFiles.toPILData()))
    # print(type(transferBase64ImageFiles.toBytes()))
    # print(type(transferBase64ImageFiles.toLocalFiles('./1.jpg')))
