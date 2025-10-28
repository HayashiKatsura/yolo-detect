import os


from _api._utils.ImagestransferComponent.FromBase64ImageFiles import TransferBase64ImageFiles
from _api._utils.ImagestransferComponent.TransferImageFilesMain import TransferImageFiles


class TransferLocalImageFiles(TransferImageFiles):
    """
    接收本地文件
    """
    def __init__(self, imageFiles):
        super().__init__(imageFiles)
        self.imageFiles = imageFiles
        self.isPdf = False
        if os.path.splitext(self.imageFiles)[1] == '.pdf':
            self.isPdf = True

    def toNdarray(self):
        import cv2
        if self.isPdf:
            return TransferBase64ImageFiles(self.imageFiles).toNdarray()
        return cv2.imread(self.imageFiles)

    def toPILData(self):
        from PIL import Image
        if self.isPdf:
            return TransferBase64ImageFiles(self.imageFiles).toPILData()
        return Image.open(self.imageFiles)

    def toBytes(self):
        if self.isPdf:
            return TransferBase64ImageFiles(self.imageFiles).toBytes()
        with open(self.imageFiles, 'rb') as f:
            return f.read()

    def toLocalFiles(self):
        pass

    def toBase64(self):
        import base64
        if self.isPdf:
            return TransferBase64ImageFiles(self.imageFiles).toBase64()
        return base64.b64encode(self.toBytes()).decode('utf-8')


if __name__ == '__main__':
    PIC_PATH = "/Users/hayashi/Documents/HL-information/DocumentRocogntioByPython/pics/bankcard/bankcard1.jpg"
    transferLocalImageFiles = TransferLocalImageFiles(PIC_PATH)
    print(type(transferLocalImageFiles.toNdarray()))
    print(type(transferLocalImageFiles.toPILData()))
    print(type(transferLocalImageFiles.toBytes()))
