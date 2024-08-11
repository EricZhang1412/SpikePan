import torch.nn.modules as nn
import torch
import cv2
import numpy as np

from snnmodel import PanNet, summaries
import h5py
import scipy.io as sio
import os
from data import Dataset_Pro
from torch.utils.data import DataLoader
from draw import *
###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################
def input_replicate(input, num):
    N, C, H, W = input.size()
    return input.unsqueeze(1).expand(N, num, C, H, W).reshape(N, num * C, H, W)

def load_set(file_path):
    data = h5py.File(file_path) 

    # lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    # ms_hp = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    # pan_hp = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    gt = torch.tensor(data["gt"][...])  
    lms = torch.tensor(data["lms"][...])
    ms = torch.tensor(data["ms"][...])
    pan = torch.tensor(data["pan"][...])

    return gt, lms, ms, pan



###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = 'D:/AI/PanNet-Code-Pytorch/WeightsT12/50.pth'   # chose model
i = 1
def test(test_data_loader):
    print('Start testing...')
    # gt, lms, ms_hp, pan_hp = load_set(file_path)
    i = 1
    model = PanNet(T=12).cuda().eval()   # fixed, important!
    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            
            gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            # pan_hp = input_replicate(pan_hp, 8)
            # model_input = pan_hp - lms
            # print(model_input.shape)

            hp_sr = model(ms_hp, pan_hp)  # call model
            sr = hp_sr + lms  # output:= lms + hp_sr

            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            file_string = f"{i}.mat"
            save_name = os.path.join("test_results", file_string) # fixed! save as .mat format that will used in Matlab!
            sio.savemat(save_name, {'test_data': sr})  # fixed!

            sr = to_rgb(sr)
            cv2.imwrite(f"test_results/{i}.png",sr*255)
            i = i + 1

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    test_set = Dataset_Pro('./training_data/valid_small.h5')
    test_data_loader = DataLoader(dataset=test_set,
                                  num_workers=0,
                                  batch_size=1,
                                  shuffle=False)
    test(test_data_loader)   # recall test function