from _api._utils.ImagestransferComponent.TransferImageFilesMain import TransferImageFiles


class TransferPILImageFiles(TransferImageFiles):
    def __init__(self, imageFiles):
        super().__init__(imageFiles)
        self.imageFiles = imageFiles

    def toNdarray(self):
        import numpy as np
        # 转换PIL图像到numpy数组
        return np.array(self.toPILData())

    def toPILData(self):
        return self.imageFiles

    def toBytes(self):
        from PIL import Image
        import io
        # 创建一个BytesIO对象
        output = io.BytesIO()
        # 将图像保存到BytesIO对象中，格式可根据需要指定，如'PNG'、'JPEG'等
        self.imageFiles.save(output, 'PNG')
        # 重置BytesIO对象的指针到起始位置，准备读取或返回
        output.seek(0)
        # 返回包含图像字节流的BytesIO对象
        return output

    def toLocalFiles(self, savePath=None):
        if savePath:
            self.toPILData().save(savePath)

    def toBase64(self):
        import base64
        return base64.b64encode(self.toBytes().getvalue()).decode('utf-8')
