"""
img = PIL.Image.open()
"""

from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

def equalize(img):
    return ImageOps.equalize(img)

def solarize(img, threshold=128):
    return ImageOps.solarize(img, threshold)

def solarize_add(img, add=30):
    img = np.array(img).astype(np.int32)
    mask = img < 128
    img[mask] = np.clip(img[mask] + add, 0, 255)
    return Image.fromarray(img.astype(np.uint8))

def adjust_contrast(img, factor=1.5):
    return ImageEnhance.Contrast(img).enhance(factor)

def adjust_color(img, factor=1.5):
    return ImageEnhance.Color(img).enhance(factor)

def adjust_brightness(img, factor=1.5):
    return ImageEnhance.Brightness(img).enhance(factor)

def adjust_sharpness(img, factor=2.0):
    return ImageEnhance.Sharpness(img).enhance(factor)
