import torch.nn.modules as nn
import torch
import cv2
import numpy as np
# from fusionnet_model import FusionNet
from spike_fusionnet_model import FusionNet, FusionNet_8_res, Inception_like_FusionNet_with_BNTT_and_mines, Inception_like_FusionNet_without_bn
import h5py
import scipy.io as sio
import os
# from data import Dataset_Pro
from torch.utils.data import DataLoader
import torch.utils.data as data
import draw
from einops import rearrange
from spikingjelly.activation_based import neuron, encoding, functional, layer, monitor


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
ckpt = 'A:/projects/fusionnet/200_tebn.pth'   # chose model

def save_mat_data(sr, scale, output_dir):
    mat_dir = os.path.join(output_dir, "results")
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    data_size = sr.shape[0]
    for i in range(data_size):
        batch = rearrange(sr[i], 'c h w -> h w c') * scale
        batch = batch.cpu().numpy()
        out_file = os.path.join(mat_dir, f'output_mulExm_{i}.mat')
        sio.savemat(out_file, {"sr": batch})

def test(test_data_loader):
    print('Start testing...')
    # gt, lms, ms_hp, pan_hp = load_set(file_path)
    i = 1
    model = Inception_like_FusionNet_with_BNTT_and_mines(8, 32, 8, T=16).cuda().eval()   # fixed, important!
    # spike_seq_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
    # for param in model.parameters():
    #     param.data.abs_()

    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    test_folder = 'test_results_Inception_like_FusionNet_with_BNTT_and_mines_200_tebn_mines'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

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

            hp_sr = model(model_input)  # call model
            sr = hp_sr + lms  # output:= lms + hp_sr

            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            file_string = f"output_mulExm_{i-1}.mat"
            # save_mat_data(sr, 2047, test_folder)
            save_name = os.path.join(test_folder, file_string) # fixed! save as .mat format that will used in Matlab!
            sio.savemat(save_name, {'sr': sr})  # fixed!

            # draw the image
            # sr = draw.linstretch(sr, 0.01, 0.99)  # fixed!
            sr = draw.to_rgb(sr)
            cv2.imwrite(test_folder + f"/{i}.png", sr*255)  # fixed! save as .png format that will used in Python!

            i = i + 1

            functional.reset_net(model)

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