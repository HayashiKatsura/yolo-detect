import glob
import PIL
import numpy as np
import imgaug.augmenters as iaa
import torch
import math
from torchvision import transforms
from torchvision.utils import save_image


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

resize = 288
imgsize = 288
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

anomaly_source_path = '/home/panxiang/Documents/kweilxfilebox/GLASS/GLASS/datasets/dtd/images'
anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg") )
image = PIL.Image.open('/home/panxiang/Documents/kweilxfilebox/ultralytics/_project/mydata/_images/single_class/_normally/000.png').convert("RGB")

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
image = transform_img(image)

aug = PIL.Image.open(np.random.choice(anomaly_source_paths)).convert("RGB")
transform_aug = rand_augmenter()
aug = transform_aug(aug)
mask_all = perlin_mask(image.shape, imgsize // 8, 0, 6, torch.tensor([1]), 1)
mask_s = torch.from_numpy(mask_all[0])
mask_l = torch.from_numpy(mask_all[1])
beta = np.random.normal(loc=mean, scale=std)
beta = np.clip(beta, .2, .8)
aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l
save_image(image, 'image.png')
save_image(aug_image, 'aug_image.png')
print() 


if __name__ == '__main__':
    pass