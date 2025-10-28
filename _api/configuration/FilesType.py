from enum import Enum

class FilesType(Enum):
    """
    文件类型
    1. images: 上传的图像
    2. weights: 训练的权重
    3. datasets: 数据集
    4. others: 其他文件
    5. compressed: 压缩文件
    6. yamls: 数据集yaml文件
    7. detect: 检测信息
    """
    images = 'images'
    videos = 'videos'
    weights = 'weights'
    datasets = 'datasets'
    others = 'others'
    compressed = 'compressed'
    yamls = 'yamls'
    detect = 'detect'
    train  = 'train'
    val    = 'val'
    

    def __str__(self):
        return str(self.value)