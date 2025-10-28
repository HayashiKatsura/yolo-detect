from _api._utils.ImagestransferComponent.TransferImageFilesMain import TransferImageFiles


class TransferBytesImageFiles(TransferImageFiles):
    """
    接收文件的字节流
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
        return Image.open(self.imageFiles)

    def toBytes(self):
        return self.imageFiles

    def toLocalFiles(self, savePath=None):
        if savePath:
            self.toPILData().save(savePath)

    def toBase64(self):
        import base64
        return base64.b64encode(self.imageFiles).decode('utf-8')

