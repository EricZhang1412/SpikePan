import torch.nn.modules as nn
import torch
import cv2
import numpy as np
# from fusionnet_model import FusionNet
from spike_fusionnet_model import FusionNet, FusionNet_8_res
import h5py
import scipy.io as sio
import os
# from data import Dataset_Pro
from torch.utils.data import DataLoader
import torch.utils.data as data
import draw

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


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=Nx8x64x64

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / 2047.
        self.gt = torch.from_numpy(gt1)  # NxCxHxW: 8x64x64

        print(self.gt.size())

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / 2047.
        self.lms = torch.from_numpy(lms1) # 8x64x64

        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3   8x16x16
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWxC
        # ms1_tmp = get_edge(ms1)  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1).permute(0, 3, 1, 2) # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        # pan_hp_tmp = get_edge(pan1)   # NxHxW
        pan_hp_tmp = np.expand_dims(pan1, axis=3)   # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2) # Nx1xHxW:

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan_hp[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = 'A:/projects/fusionnet/Weight_spike_FusionNet_64_mean_a100/350.pth'   # chose model
i = 1
def test(test_data_loader):
    print('Start testing...')
    # gt, lms, ms_hp, pan_hp = load_set(file_path)
    i = 1
    model = FusionNet(8, 64, 8, T=16).cuda().eval()   # fixed, important!
    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            
            gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            pan_hp = input_replicate(pan_hp, 8)
            model_input = pan_hp - lms
            print(model_input.shape)

            # hp_sr = model(model_input)  # call model
            # sr = hp_sr + lms  # output:= lms + hp_sr

            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            gt = torch.squeeze(gt).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
            file_string = f"{i}_gt.mat"
            save_name = os.path.join("test_results_inception_like_fusionnet_with_BNTT", file_string) # fixed! save as .mat format that will used in Matlab!
            sio.savemat(save_name, {'test_data_gt': gt})  # fixed!

            # draw the image
            # sr = draw.linstretch(sr, 0.01, 0.99)  # fixed!
            gt = draw.to_rgb(gt)
            cv2.imwrite(f"test_results_inception_like_fusionnet_with_BNTT/{i}_gt.png", gt*255)  # fixed! save as .png format that will used in Python!

            i = i + 1

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    test_set = Dataset_Pro('A:/pancollection/wv3/test_wv3_multiExm1.h5')
    test_data_loader = DataLoader(dataset=test_set,
                                  num_workers=0,
                                  batch_size=1,
                                  shuffle=False)
    test(test_data_loader)   # recall test function