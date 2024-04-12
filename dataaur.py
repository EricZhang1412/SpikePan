import random
from torchvision import transforms as T
import torch
# 旋转
def rotate(gt, lms, ms_hp, pan_hp):
    # Randomly choose rotation angle between -degree and degree
    degree = random.uniform(0, 10)  # Adjust the degree range as needed

    # Create a single random rotation transformation for all tensors
    rotate_transform = T.RandomRotation(degrees=degree, fill=(0,))  # Specify fill value for rotation

    # Apply rotation to all tensors
    gt = rotate_transform(gt)
    lms = rotate_transform(lms)
    ms_hp = rotate_transform(ms_hp)
    pan_hp = rotate_transform(pan_hp)
    return gt, lms, ms_hp, pan_hp

def crop(gt, lms, ms_hp, pan_hp, crop_min=0.8, crop_max=1.0):
  if crop_min < crop_max:
    # Adjust crop range based on your needs
    crop_ratio = random.uniform(crop_min, crop_max)
    new_h = int(crop_ratio * gt.shape[2])
    new_w = int(crop_ratio * gt.shape[3])
    i, j, h, w = T.RandomCrop.get_params(gt, output_size=(new_h, new_w))
    gt = gt[:, :, i:i+h, j:j+w]
    lms = lms[:, :, i:i+h, j:j+w]
    ms_hp = ms_hp[:, :, i:i+h, j:j+w]
    pan_hp = pan_hp[:, :, i:i+h, j:j+w]
  else:
    raise ValueError("crop_min must be less than crop_max")
  return gt, lms, ms_hp, pan_hp

def flip(gt, lms, ms_hp, pan_hp):
    if random.random() > 0.5:  # 以50%的概率进行水平翻转
        gt = torch.flip(gt, dims=[3])  # 在宽度维度上进行翻转
        lms = torch.flip(lms, dims=[3])
        ms_hp = torch.flip(ms_hp, dims=[3])
        pan_hp = torch.flip(pan_hp, dims=[3])
    return gt, lms, ms_hp, pan_hp
def cutout(gt, lms, ms_hp, pan_hp, prob=0.5, size=8):
    if random.random() < prob:
        h, w = gt.shape[2], gt.shape[3]
        y = random.randint(0, h - size - 1)
        x = random.randint(0, w - size - 1)
        gt[:, :, y:y+size, x:x+size] = 0  # 将区域置零
        lms[:, :, y:y+size, x:x+size] = 0
        ms_hp[:, :, y:y+size, x:x+size] = 0
        pan_hp[:, :, y:y+size, x:x+size] = 0
    return gt, lms, ms_hp, pan_hp 
    
def dataaug(gt, lms, ms_hp, pan_hp,ro = True, cr = True, fl = True, cut = True):
    if ro == True:
        gt, lms, ms_hp, pan_hp = rotate(gt, lms, ms_hp, pan_hp)
    if cr == True:
        gt, lms, ms_hp, pan_hp = crop(gt, lms, ms_hp, pan_hp, crop_min=0.8, crop_max=1.0)
    if fl == True:
        gt, lms, ms_hp, pan_hp = flip(gt, lms, ms_hp, pan_hp)
    if cut == True:
        gt, lms, ms_hp, pan_hp = cutout(gt, lms, ms_hp, pan_hp, prob=0.5, size=8)
    return gt, lms, ms_hp, pan_hp
