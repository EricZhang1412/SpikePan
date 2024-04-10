import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from spike_fusionnet_model import input_replicate, summaries, Dataset_Pro

from net_struct import \
Inception_SFusionNet_with_SEW_RDB_tEBN_2_paths, \
Inception_SFusionNet_with_SEW_tEBN_2_paths, \
Inception_SFusionNet_with_tEBN_2_paths, \
Inception_SFusionNet_with_tEBN_BN, \
Inception_SFusionNet_ori_with_TEBN_2_ATAN, \
Inception_SFusionNet_ori_with_TEBN_bn_ATAN, \
Inception_SFusionNet_ori_with_TEBN_2_ATAN_wT



import numpy as np
from quality_indices import SAM_group
import scipy.io as sio
from torchvision import transforms
from PIL import Image

import shutil
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based import neuron, encoding, functional


SEED = 3407
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False
# args = get_args_parser().parse_args()

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 1e-5  #学习率
epochs = 1000 # 450
ckpt = 50
batch_size = 4
T1 = 4
T2 = 2

model_name = 'Inception_SFusionNet_ori_with_TEBN_2_ATAN_wT'
# model_path = "Weights/250.pth"
model_path = ''
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
# model = FusionNet(16, 32, 8).cuda()
# with lasso
# model = Inception_SFusionNet_ori_with_TEBN_2_ATAN.FusionNet(8, 32, 8, T=T).cuda()
model = Inception_SFusionNet_ori_with_TEBN_2_ATAN_wT.FusionNet(8, 32, 8, T1=T1,T2=T2).cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('FusionNet is Successfully Loaded from %s' % (model_path))

summaries(model, grad=True)    ## Summary the Network
# criterion = nn.MSELoss().cuda()  ## Define the Loss function L1Loss
criterion = nn.L1Loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)   # learning-rate update

def save_checkpoint(model, epoch):  # save model function
    model_folder = 'Weight_' + model_name
    model_out_path = model_folder + '/' + "{}.pth".format(epoch)
    # create file folder if it is not existed
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')

    i = 5
    
    best_loss = 1
    best_epoch = 0
    best_validate_loss = 1
    best_validate_epoch = 0
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        epoch_sam_value = []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            pan_hp = input_replicate(pan_hp, 8)  # replicate the pan image to 8 channels

            model_input = pan_hp - lms
            optimizer.zero_grad()  # fixed
            hp_sr = model(model_input)  # call model
            sr = hp_sr + lms  # output:= lms + hp_sr

            loss = criterion(sr, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()   # fixed
            optimizer.step()  # fixed
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(model)

            # for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                # writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

            #lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        if t_loss<best_loss:
            best_loss = t_loss
            best_epoch = epoch
            
        # writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        print('best_Epoch: {}/{} best_training loss: {:.7f}'.format(epochs, best_epoch, best_loss)) 

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()# fixed
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
                pan_hp = input_replicate(pan_hp, 8)  # replicate the pan image to 8 channels
                # model_eval_input = torch.cat((lms, pan_hp), 1)  # concatenate ms_hp and pan_hp
                model_eval_input = pan_hp - lms

                hp_sr = model(model_eval_input)
                sr = hp_sr + lms

                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())
                
                functional.reset_net(model)
                
        if epoch % 5 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            if v_loss<best_validate_loss:
                best_validate_loss=v_loss
                best_validate_epoch = epoch
            # writer.add_scalar('val_Fusionnet_64_mean/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             best validate epoch: {} best validate loss: {:.7f}'.format(best_validate_epoch, best_validate_loss))
            ######add image output
            ######pick one image to show
            sr_img_sample = sr[0].cpu().detach()
            sr_img_sample_ch_R = sr_img_sample[4]
            sr_img_sample_ch_G = sr_img_sample[2]
            sr_img_sample_ch_B = sr_img_sample[1]

            sr_img_sample_RGB = torch.stack([sr_img_sample_ch_R, sr_img_sample_ch_G, sr_img_sample_ch_B], dim=0)
            
            sr_img = transforms.ToPILImage()(sr_img_sample_RGB)
            # save to file
            
            image_folder = 'val_' + model_name
            image_path = image_folder + '/' + str(i) + '.png'
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            sr_img.save(image_path)
            sample_img = Image.open(image_path)

            # to float
            sample_img_array = np.array(sample_img).astype(np.float32)
            
            # writer.add_image(image_path, sample_img_array, 1, dataformats='HWC')
            i = i + 5
    

    # writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('D:/AI/pancollection/wv3/train_wv3.h5')  # creat data for training   # 100
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,  # 3 32
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('D:/AI/pancollection/wv3/valid_wv3.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
    

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
