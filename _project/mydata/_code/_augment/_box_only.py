import numpy as np
from PIL import Image, ImageOps
def bbox_only_equalize(img, bboxes):
    img_np = np.array(img)
    for x1, y1, x2, y2 in bboxes:
        region = Image.fromarray(img_np[y1:y2, x1:x2])
        region = ImageOps.equalize(region)
        img_np[y1:y2, x1:x2] = np.array(region)
    return Image.fromarray(img_np)

def bbox_only_flip_lr(img, bboxes):
    img_np = np.array(img)
    for x1, y1, x2, y2 in bboxes:
        region = Image.fromarray(img_np[y1:y2, x1:x2])
        region = region.transpose(Image.FLIP_LEFT_RIGHT)
        img_np[y1:y2, x1:x2] = np.array(region)
    return Image.fromarray(img_np)

def bbox_only_translate_y(img, bboxes, offset=10):
    img_np = np.array(img)
    h = img_np.shape[0]
    for x1, y1, x2, y2 in bboxes:
        region = img_np[y1:y2, x1:x2].copy()
        if y1 + offset < 0 or y2 + offset > h:
            continue  # 防止越界
        img_np[y1+offset:y2+offset, x1:x2] = region
    return Image.fromarray(img_np)
