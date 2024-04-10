import torch.nn.modules as nn
import torch
import cv2
import numpy as np
# from fusionnet_model import FusionNet
from net_struct import \
    Inception_SFusionNet_with_SEW_RDB_tEBN_2_paths, \
    Inception_SFusionNet_with_SEW_tEBN_2_paths, \
    Inception_SFusionNet_with_tEBN_2_paths, \
    Inception_SFusionNet_with_tEBN_BN, \
    Inception_SFusionNet_ori_with_TEBN_2_ATAN
import h5py
import scipy.io as sio
import os
# from data import Dataset_Pro
from torch.utils.data import DataLoader
import torch.utils.data as data
import draw
from einops import rearrange
from spikingjelly.activation_based import neuron, encoding, functional, layer, monitor

from spikingjelly import visualizing
from matplotlib import pyplot as plt
from FSDS_code import FrequencySpectrumDistributionSimilarity
from pytorch_model_summary import summary


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
        return self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan_hp[index, :, :, :].float()

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################
ckpt = 'A:/projects/fusionnet/130_t_32.pth'   # chose model
T = 32
connect_type = 'add'

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
    s_list = []
    v_list = []
    model = Inception_SFusionNet_ori_with_TEBN_2_ATAN.FusionNet(8, 32, 8, T=T).cuda().eval()   # fixed, important!
    # spike_seq_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
    # for param in model.parameters():
    #     param.data.abs_()
    

    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    test_folder = 'test_results_multiExm1_130_t_32'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    with torch.no_grad():
        for iteration, batch in enumerate(test_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            
            lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            pan_hp = input_replicate(pan_hp, 8)
            model_input = pan_hp - lms
            print(model_input.shape)
            print(summary(model, model_input, show_input=False, show_hierarchical=False))

            hp_sr, lif1_out = model(model_input)  # call model
            sr = hp_sr + lms  # output:= lms + hp_sr

            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            # file_string = f"output_mulExm_{i-1}.mat"
            # save_mat_data(sr, 2047, test_folder)
            # save_name = os.path.join(test_folder, file_string) # fixed! save as .mat format that will used in Matlab!
            # sio.savemat(save_name, {'sr': sr})  # fixed!

            # draw the image
            # sr = draw.linstretch(sr, 0.01, 0.99)  # fixed!
            # sr = draw.to_rgb(sr)
            # cv2.imwrite(test_folder + f"/{i}.png", sr*255)  # fixed! save as .png format that will used in Python!

            # calculate FSDS value
            # fsds_result = FrequencySpectrumDistributionSimilarity(sr, gt)
            # print(f"fsds_result: {fsds_result}")
            

            #print spike map using spikingjelly.visualizing
            # spike_seq_monitor = monitor.OutputMonitor(model.lif1, neuron.LIFNode)
            # v_seq_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=model, instance=model.lif1)
            # s_list.append(spike_seq_monitor.records)
            # v_list.append(model.lif1.v_seq.cpu().numpy())
            # s_list = torch.tensor(s_list)
            # v_list = torch.cat(model.lif1.v_seq)

            # visualizing.plot_1d_spikes(spikes=np.asarray(model.lif1.v_seq.unsqueeze(0).cpu()), 
            #                            title='Membrane Potentials', 
            #                            xlabel='Simulating Step',
            #                            ylabel='Neuron Index', 
            #                            dpi=200)
            # visualizing.plot_2d_heatmap(array=np.asarray(lif1_out.flatten(1,4).cpu()), 
            #                             title='Membrane Potentials', 
            #                             xlabel='Simulating Step',
            #                             ylabel='Neuron Index', 
            #                             int_x_ticks=True, 
            #                             x_max=T,
            #                             dpi=200)
            # # plt.show()
            # plt.savefig(f'membrane_potentials_model.lif1_out_{i}.png')
            i = i + 1

            functional.reset_net(model)
            del sr, hp_sr, lif1_out, model_input, lms, ms_hp, pan_hp
            torch.cuda.empty_cache()


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