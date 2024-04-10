import torch
import torch.nn as nn
import numpy as np
import math
from spikingjelly.activation_based import base, neuron, encoding, functional, surrogate, layer

import torch.nn.init as int
import h5py
import torch.utils.data as data
from torch.nn.modules.batchnorm import _BatchNorm
import einops

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
    

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, T:int):
        super(Resblock, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        # functional.set_step_mode(self, step_mode='m')


    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x_identity
        out = self.lif2(out)
        return out
    
class FusionNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, T:int):
        super(FusionNet, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.resblock1 = Resblock(mid_ch, mid_ch, T=self.T)
        self.resblock2 = Resblock(mid_ch, mid_ch, T=self.T)
        self.resblock3 = Resblock(mid_ch, mid_ch, T=self.T)
        self.resblock4 = Resblock(mid_ch, mid_ch, T=self.T)
        self.conv2 = layer.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        self.bn3 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=mid_ch)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv3 = layer.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        lif1_out = x
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
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
        return output, lif1_out
