import glob
from turtle import st
import PIL
import numpy as np
import imgaug.augmenters as iaa
import torch
import math
from torchvision import transforms
from torchvision.utils import save_image
from scipy import ndimage
import cv2
import os

# Import original functions
def generate_thr(img_shape, min=0, max=4):
    min_perlin_scale = min
    max_perlin_scale = max
    perlin_scalex = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scaley = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_noise_np = rand_perlin_2d_np((img_shape[1], img_shape[2]), (perlin_scalex, perlin_scaley))
    threshold = 0.5
    perlin_noise_np = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise_np)
    perlin_thr = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
    return perlin_thr


def perlin_mask(img_shape, feat_size, min, max, mask_fg, flag=0):
    """
    mask_s：下采样后的低分辨率掩膜（用于模型监督）
    mask_l：原始分辨率的高斯噪声掩膜（用于图像混合）
    """
    mask = np.zeros((feat_size, feat_size))
    while np.max(mask) == 0:
        perlin_thr_1 = generate_thr(img_shape, min, max)
        perlin_thr_2 = generate_thr(img_shape, min, max)
        temp = torch.rand(1).numpy()[0]
        if temp > 2 / 3: # 加法模式
            perlin_thr = perlin_thr_1 + perlin_thr_2
            perlin_thr = np.where(perlin_thr > 0, np.ones_like(perlin_thr), np.zeros_like(perlin_thr))
        elif temp > 1 / 3: # 乘法模式
            perlin_thr = perlin_thr_1 * perlin_thr_2
        else: # 单噪声模式
            perlin_thr = perlin_thr_1
        perlin_thr = torch.from_numpy(perlin_thr)
        perlin_thr_fg = perlin_thr * mask_fg
        down_ratio_y = int(img_shape[1] / feat_size)
        down_ratio_x = int(img_shape[2] / feat_size)
        mask_ = perlin_thr_fg
        mask = torch.nn.functional.max_pool2d(perlin_thr_fg.unsqueeze(0).unsqueeze(0), (down_ratio_y, down_ratio_x)).float()
        mask = mask.numpy()[0, 0]
    mask_s = mask
    if flag != 0:
        mask_l = mask_.numpy()
    if flag == 0:
        return mask_s
    else:
        return mask_s, mask_l


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1],
                                                  axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


# New function to extract noise patch positions in YOLO format
def extract_noise_patches_yolo(mask, class_id=0):
    """
    Extract bounding boxes of noise patches in YOLO format.
    
    Args:
        mask: Binary mask where 1 indicates noise regions
        class_id: Class ID for the noise patches (default: 0)
        
    Returns:
        List of bounding boxes in YOLO format: [class_id, center_x, center_y, width, height]
    """
    # Convert to numpy array if it's a torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Label connected components
    labeled_mask, num_features = ndimage.label(binary_mask)
    
    # Get bounding boxes for each labeled region
    yolo_boxes = []
    img_height, img_width = mask.shape
    
    # For each connected component (noise patch)
    for i in range(1, num_features + 1):
        # Get coordinates of pixels belonging to this component
        component_coords = np.where(labeled_mask == i)
        
        if len(component_coords[0]) == 0:
            continue
            
        # Calculate bounding box
        y_min, y_max = np.min(component_coords[0]), np.max(component_coords[0])
        x_min, x_max = np.min(component_coords[1]), np.max(component_coords[1])
        
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        
        # Skip very small regions (likely noise)
        min_size = 5  # Minimum size in pixels
        if width < min_size or height < min_size:
            continue
            
        # Convert to YOLO format (normalized center_x, center_y, width, height)
        center_x = (x_min + x_max) / 2.0 / img_width
        center_y = (y_min + y_max) / 2.0 / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        
        # Add to boxes list
        yolo_boxes.append([class_id, center_x, center_y, norm_width, norm_height])
    
    return yolo_boxes

# Function to save YOLO format annotations to text file
def save_yolo_annotations(boxes, filename):
    """
    Save YOLO format annotations to a text file.
    
    Args:
        boxes: List of bounding boxes in YOLO format [class_id, center_x, center_y, width, height]
        filename: Name of the output text file
    """
    with open(filename, 'w') as f:
        for box in boxes:
            # Format: class_id center_x center_y width height
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

# Function to visualize bounding boxes on the image
def visualize_boxes(image, boxes, output_path):
    """
    Visualize bounding boxes on the image and save it.
    
    Args:
        image: Tensor image (C, H, W)
        boxes: List of bounding boxes in YOLO format [class_id, center_x, center_y, width, height]
        output_path: Path to save the visualization
    """
    # Convert tensor to numpy image for OpenCV
    if isinstance(image, torch.Tensor):
        # Denormalize if needed
        img_np = image.clone().detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = np.array(image)
    
    img_height, img_width = img_np.shape[:2]
    
    # Convert back to BGR for OpenCV
    if img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw boxes
    for box in boxes:
        class_id, center_x, center_y, width, height = box
        
        # Convert normalized coordinates to pixel values
        center_x_px = int(center_x * img_width)
        center_y_px = int(center_y * img_height)
        width_px = int(width * img_width)
        height_px = int(height * img_height)
        
        # Calculate top-left and bottom-right coordinates
        x1 = int(center_x_px - width_px / 2)
        y1 = int(center_y_px - height_px / 2)
        x2 = int(center_x_px + width_px / 2)
        y2 = int(center_y_px + height_px / 2)
        
        # Draw rectangle
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Noise {int(class_id)}"
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save image
    cv2.imwrite(output_path, img_np)


# Example usage (modified from original code)
# Add functions to generate random augmentation
resize = imgsize = 640
mean = 0.5
std = 0.1
brightness_factor = 0
contrast_factor = 0
saturation_factor = 0
h_flip_p = v_flip_p = gray_p = rotate_degrees = translate = scale = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def rand_augmenter():    
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)), # 随机调整对比度
            transforms.ColorJitter(brightness=(0.8, 1.2)), # 随机调整亮度
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),# 调整饱和度和色度
            transforms.RandomHorizontalFlip(p=1), # 100%水平翻转
            transforms.RandomVerticalFlip(p=1), # 100%垂直翻转
            transforms.RandomGrayscale(p=1), # 100%灰度化
            transforms.RandomAutocontrast(p=1), # 100%自动对比度
            transforms.RandomEqualize(p=1), # 100%直方图均衡化
            transforms.RandomAffine(degrees=(-45, 45)), # 随机旋转
        ]
        # 随机选取3个不重复的增强操作
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(resize), # 调整图像尺寸
            list_aug[aug_idx[0]], # 第一个增强操作
            list_aug[aug_idx[1]], # 第二个增强操作
            list_aug[aug_idx[2]], # 第三个增强操作
            transforms.CenterCrop(imgsize), # 中心裁剪
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # 标准化
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

if __name__ == '__main__':
    # 合成纹理特征
    anomaly_source_path = '/home/panxiang/Documents/kweilxfilebox/GLASS/GLASS/datasets/dtd/images'  # 替换为您的纹理数据集路径
    # 正常样本文件夹路径
    source_folder = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/label_and_test_images/test_images/single_class/synthetic_200_normal'
    _base_floder = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_a_datasets/single_class/perlin_noise/perlin_noise_200pics'
    
    # mask 文件夹路径
    mask_folder = os.path.join(_base_floder,'mask')
    # 标注可视化
    visualize_folder = os.path.join(_base_floder,'visualize')
    # 训练集划分比例
    train_val_rate = 0.3
    train_folder = os.path.join(_base_floder,'images','train')
    val_folder = os.path.join(_base_floder,'images','val')
    _labels_train_folder = os.path.join(_base_floder,'labels','train')
    _labels_val_folder = os.path.join(_base_floder,'labels','val')
    os.makedirs(mask_folder,exist_ok=True)
    os.makedirs(visualize_folder,exist_ok=True)
    os.makedirs(train_folder,exist_ok=True)
    os.makedirs(val_folder,exist_ok=True)
    os.makedirs(_labels_train_folder,exist_ok=True)
    os.makedirs(_labels_val_folder,exist_ok=True)
    
    # 合成样本总数 iter_count * len(source_folder)
    iter_count = 1
    FILE_NAME = 0
    for i in range(iter_count):
        FILE_COUNT = 0
        for files in os.listdir(source_folder):
            if str(files).endswith('.png'):
                _file_name = str(files).split('.')[0]
                image_path = os.path.join(source_folder, files)
                
                # image_path = '/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_images/single_class/_normally/000.png'  # 替换为您的正常图像路径     
                try:
                    # 加载异常纹理路径
                    anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
                    if not anomaly_source_paths:  # 如果没有找到纹理图像
                        print("未找到纹理图像，使用随机生成的图像替代")
                        anomaly_source_paths = ['dummy']
                except:
                    print("纹理路径加载错误，使用随机生成的图像替代")
                    anomaly_source_paths = ['dummy']
                
                # 图像变换
                transform_img = [
                    transforms.Resize(resize),
                    transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
                    transforms.RandomHorizontalFlip(h_flip_p),
                    transforms.RandomVerticalFlip(v_flip_p),
                    transforms.RandomGrayscale(gray_p),
                    transforms.RandomAffine(rotate_degrees,
                                        translate=(translate, translate),
                                        scale=(1.0 - scale, 1.0 + scale),
                                        interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(imgsize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
                transform_img = transforms.Compose(transform_img)
                
                # 加载原图
                try:
                    image = PIL.Image.open(image_path).convert("RGB")
                    image = transform_img(image)
                except:
                    raise ValueError("图像加载错误")
                
                # 加载纹理
                try:
                    aug_path = np.random.choice(anomaly_source_paths)
                    if aug_path == 'dummy':
                        aug = torch.randn(3, imgsize, imgsize)  # 生成随机图像
                    else:
                        aug = PIL.Image.open(aug_path).convert("RGB")
                        transform_aug = rand_augmenter()
                        aug = transform_aug(aug)
                except:
                    raise ValueError("纹理图像加载错误")
          
                
                # 生成噪声掩码
                mask_all = perlin_mask(image.shape, imgsize // 8, 0, 6, torch.tensor([1]), 1)
                mask_s = torch.from_numpy(mask_all[0])
                mask_l = torch.from_numpy(mask_all[1])
                
                # 提取噪声斑块的YOLO格式位置信息
                noise_boxes = extract_noise_patches_yolo(mask_l)
                
                # 生成合成图像
                beta = np.random.normal(loc=mean, scale=std)
                beta = np.clip(beta, .2, .8)
                aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l
                
                # if(_p:=torch.rand(1).item()>0.3):
                if FILE_COUNT % 3 == 0: # 接近0.3的概率保存到测试集
                    save_folder = val_folder
                else:
                    save_folder = train_folder
                
                # 保存YOLO格式的标注 labels
                save_yolo_annotations(noise_boxes, os.path.join(str(save_folder).replace('images', 'labels'),f"{str(FILE_NAME).zfill(3)}_{_file_name}.txt"))
                
                # 保存原始图像和合成图像
                image_display = image.clone()
                for t, m, s in zip(image_display, IMAGENET_MEAN, IMAGENET_STD):
                    t.mul_(s).add_(m)
                image_display.clamp_(0, 1)
                
                aug_image_display = aug_image.clone()
                for t, m, s in zip(aug_image_display, IMAGENET_MEAN, IMAGENET_STD):
                    t.mul_(s).add_(m)
                aug_image_display.clamp_(0, 1)
                
                # 保存图像
                # save_image(image_display, 'original_image.png')
                save_image(aug_image_display, os.path.join(save_folder,f"{str(FILE_NAME).zfill(3)}_{_file_name}.png"))
                save_image(mask_l.float(), os.path.join(mask_folder,f"{str(FILE_NAME).zfill(3)}_{_file_name}_mask.png"))

                # 可视化边界框
                visualize_boxes(aug_image_display, noise_boxes, os.path.join(visualize_folder,f"{str(FILE_NAME).zfill(3)}_{_file_name}_boxes.png"))
                
                FILE_NAME += 1
                FILE_COUNT += 1
                # 打印检测到的噪声区域信息
                # print(f"检测到 {len(noise_boxes)} 个噪声区域")
                # for i, box in enumerate(noise_boxes):
                #     print(f"噪声区域 {i+1}: 类别={int(box[0])}, 中心x={box[1]:.4f}, 中心y={box[2]:.4f}, 宽度={box[3]:.4f}, 高度={box[4]:.4f}")