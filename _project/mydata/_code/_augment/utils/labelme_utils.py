import json
from PIL import Image, ImageDraw
import numpy as np

def get_largest_region_and_crop(img_path, json_path):
    img = Image.open(img_path).convert("RGB")
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    max_area = 0
    largest_bbox = None
    largest_mask = None

    for shape in label_data['shapes']:
        points = [(int(p[0]), int(p[1])) for p in shape['points']]  # 转整数
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        area = (x_max - x_min) * (y_max - y_min)

        if area > max_area:
            max_area = area
            largest_bbox = (x_min, y_min, x_max, y_max)

            # 生成 mask
            mask = Image.new('L', img.size, 0)
            ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
            largest_mask = mask

    if largest_bbox:
        img_np = np.array(img)
        mask_np = np.array(largest_mask)
        # 抠图
        masked_img = img_np * mask_np[:, :, None]
        x_min, y_min, x_max, y_max = map(int, largest_bbox)
        cropped = masked_img[y_min:y_max, x_min:x_max, :]
        return Image.fromarray(cropped)
    else:
        return None
