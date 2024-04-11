import torch
import torch.nn as nn
import numpy as np
import math
from spikingjelly.activation_based import base, neuron, encoding, functional, surrogate, layer

import torch.nn.init as int
import h5py
import torch.utils.data as data
from torch.nn.modules.batchnorm import _BatchNorm

##################add tdBN##################
class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """
    def __init__(self, T: int) -> None:
        super().__init__()
        n_steps = T  ## 这里是T

        arr = torch.arange(n_steps-1,-1,-1)
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        out = torch.sum(x*self.coef, dim=-1)
        return out
    
class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, v_th=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.v_th = v_th

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([1, 2, 3, 4])
            # use biased var in train
            var = input.var([1, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.v_th * (input - mean[:, None, None, None, None]) / (torch.sqrt(var[:, None, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[:, None, None, None, None] + self.bias[:, None, None, None, None]

        return input
    

class Resblock_con5_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T):
        super(Resblock_con5_with_BNTT_and_mines, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x_identity
        out = self.lif2(out)
        out_l1_penalty_2 = torch.norm(out, 1)
        return out, out_l1_penalty_1, out_l1_penalty_2
    
class Resblock_con3_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T):
        super(Resblock_con3_with_BNTT_and_mines, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x_identity
        out = self.lif2(out)
        out_l1_penalty_2= torch.norm(out, 1)
        return out, out_l1_penalty_1, out_l1_penalty_2
    
class FusionNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, T:int):
        super(FusionNet, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_ch, mid_ch, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv_3 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        self.resblock1 = Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock2 = Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock3 = Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock4 = Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)

        self.resblock_1 = Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock_2 = Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock_3 = Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)
        self.resblock_4 = Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T)

        self.conv2 = layer.Conv2d(mid_ch * 2, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = layer.Conv2d(mid_ch * 2, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv_31 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = layer.BatchNorm2d(out_ch)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        self.bn3 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True)
        self.lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True)
        self.membrane_output_layer = MembraneOutputLayer(T=self.T)
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x_l1_penalty_1 = torch.norm(x, 1)
        x, x_res1_l1_penalty_1, x_res1_l1_penalty_2 = self.resblock1(x)
        x, x_res2_l1_penalty_1, x_res2_l1_penalty_2 = self.resblock2(x)
        x, x_res3_l1_penalty_1, x_res3_l1_penalty_2 = self.resblock3(x)
        x, x_res4_l1_penalty_1, x_res4_l1_penalty_2 = self.resblock4(x)

        x1, x1_res1_l1_penalty_1, x1_res1_l1_penalty_2  = self.resblock_1(x)
        x1, x1_res2_l1_penalty_1, x1_res2_l1_penalty_2= self.resblock_2(x1)
        x1, x1_res3_l1_penalty_1, x1_res3_l1_penalty_2= self.resblock_3(x1)
        x1, x1_res4_l1_penalty_1, x1_res4_l1_penalty_2= self.resblock_4(x1)

        x = torch.cat([x, x1], dim=2)
        # print(x.shape)
        
        x_mines = self.conv3(x)
        x_mines = self.bn3(x_mines)
        x_mines = self.lif3(x_mines)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        
        result_mines = self.lif3.v_seq.permute(1, 2, 3, 4, 0)
        result = self.lif2.v_seq.permute(1, 2, 3, 4, 0) # [N,C,H,W,T]
        result = torch.cat([result, -result_mines], dim=-1)# [N,C,H,W,2T]
        result = torch.mean(result, dim=-1)
        output = torch.tanh(result)
        l1_penalty = x_res1_l1_penalty_1 + x_res1_l1_penalty_2 + \
                    x_res2_l1_penalty_1 + x_res2_l1_penalty_2 + \
                    x_res3_l1_penalty_1 + x_res3_l1_penalty_2 + \
                    x_res4_l1_penalty_1 + x_res4_l1_penalty_2 + \
                    x1_res1_l1_penalty_1 +  x1_res1_l1_penalty_2 + \
                    x1_res2_l1_penalty_1 +  x1_res2_l1_penalty_2 + \
                    x1_res3_l1_penalty_1 +  x1_res3_l1_penalty_2 + \
                    x1_res4_l1_penalty_1 +  x1_res4_l1_penalty_2
        return output, l1_penalty