# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import Dataset_Pro
# from model import PanNet, summaries
from spannet_model import PanNet, summaries
import numpy as np
from torchvision import transforms
from PIL import Image
import shutil
from torch.utils.tensorboard import SummaryWriter
import argparse
from spikingjelly.activation_based import neuron, encoding, functional
from quality_indices import SAM_group
###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
parser = argparse.ArgumentParser(description='pannet')
parser.add_argument('-T', default=32, type=int, help='simulating time-steps')
parser.add_argument('-device', default='cuda:0', help='device')
args = parser.parse_args()
# print(args)
lr = 0.001  #学习率
epochs = 500 # 450
ckpt = 20
batch_size = 4
model_path = ''

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = PanNet(T=args.T).cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('PANnet is Successfully Loaded from %s' % (model_path))

summaries(model, grad=True)    ## Summary the Network
# criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function L2Loss
criterion = nn.L1Loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   ## optimizer 1: Adam
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)   # learning-rate update
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)  
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.1)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180

# ============= 4) Tensorboard_show + Save_model ==========#
# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
#   shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs

writer = SummaryWriter('./train_logs/pannet')    ## Tensorboard_show: case 2

# def save_checkpoint(model, epoch):  # save model function
#     model_out_path = 'WeightsT12' + '/' + "{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
def save_checkpoint(model, epoch):  # save model function
    model_folder = 'Weight_' + 'Inception_SFusionNet_ori_with_tdBN_2_ATAN'
    model_out_path = model_folder + '/' + "{}.pth".format(epoch+50)
    # create file folder if it is not existed
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')
    # epoch 450, 450*550 / 2 = 123750 / 8806 = 14/per imgae
    i = 5
    best_loss = 1
    best_epoch = 0
    best_validate_loss = 1
    best_validate_epoch = 0
    for epoch in range(start_epoch, epochs, 1):
        # start_time = time.time()
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1): # 100  3
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()

            optimizer.zero_grad()  # fixed
            # print(lms.shape)
            hp_sr = model(ms_hp, pan_hp)  # call model

            sr = lms + hp_sr  # output:= lms + hp_sr

            loss = criterion(sr, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()   # fixed
            optimizer.step()  # fixed
            functional.reset_net(model)

            for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)

        #lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        if t_loss<best_loss:
            best_loss = t_loss
            best_epoch = epoch
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        print('best_Epoch: {}/{} best_training loss: {:.7f}'.format(epochs, best_epoch, best_loss)) 
        
        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()# fixed
        with torch.no_grad():  # fixed
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, ms_hp, pan_hp = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()

                hp_sr = model(ms_hp, pan_hp)
                sr = lms + hp_sr

                loss = criterion(sr, gt)
                epoch_val_loss.append(loss.item())
                functional.reset_net(model)


        if epoch % 5 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            if v_loss<best_validate_loss:
                best_validate_loss = v_loss
                best_validate_epoch = epoch
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             best validate epoch: {} best validate loss: {:.7f}'.format(best_validate_epoch, best_validate_loss))
            i = i + 5

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('D:/AI/pancollection/gf2/train_gf2.h5')  # creat data for training   # 100
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,  # 3 32
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('D:/AI/pancollection/gf2/valid_gf2.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
