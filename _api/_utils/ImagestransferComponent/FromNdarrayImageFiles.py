from _api._utils.ImagestransferComponent.TransferImageFilesMain import TransferImageFiles


class TransferNdarrayImageFiles(TransferImageFiles):
    def __init__(self, imageFiles, reverse=True):
        super().__init__(imageFiles)
        self.imageFiles = imageFiles
        self.reverse = reverse

    def toNdarray(self):
        return self.imageFiles

    def toPILData(self):
        from PIL import Image
        # thisImage = Image.fromarray(self.imageFiles, 'RGB')
        # rgb_array = self.imageFiles[:, :, ::-1]  # 这将BGR颜色通道翻转为RGB
        # thisImage = Image.fromarray(rgb_array, 'RGB')
        # thisImage.show()
        if self.reverse:
            rgb_array = self.imageFiles[:, :, ::-1]
        else:
            rgb_array = self.imageFiles
        return Image.fromarray(rgb_array, 'RGB')
        # return Image.fromarray(self.imageFiles, 'RGBA')

    def toBytes(self):
        from PIL import Image
        import io
        # 创建一个BytesIO对象
        output = io.BytesIO()
        # 将图像保存到BytesIO对象中，格式可根据需要指定，如'PNG'、'JPEG'等
        self.toPILData().save(output, 'PNG')
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
