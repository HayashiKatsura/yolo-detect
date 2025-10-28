from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
def shear_x(img, level=0.3):
    return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

def shear_y(img, level=0.3):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

def translate_x(img, offset=50):
    return img.transform(img.size, Image.AFFINE, (1, 0, offset, 0, 1, 0))

def translate_y(img, offset=50):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, offset))

def rotate(img, angle=15):
    return img.rotate(angle, resample=Image.BILINEAR)

import numpy as np
def cutout(img, size=60):
    h, w = img.size
    x = np.random.randint(w)
    y = np.random.randint(h)
    x1 = np.clip(x - size // 2, 0, w)
    y1 = np.clip(y - size // 2, 0, h)
    x2 = np.clip(x + size // 2, 0, w)
    y2 = np.clip(y + size // 2, 0, h)
    img = np.array(img)
    img[y1:y2, x1:x2] = 127  # 灰色遮挡
    return Image.fromarray(img)
