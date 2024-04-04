import torch
import torch.nn as nn
import numpy as np
import math
from spikingjelly.activation_based import base, neuron, encoding, functional, surrogate, layer

import torch.nn.init as int
import h5py
import torch.utils.data as data
from torch.nn.modules.batchnorm import _BatchNorm

##################add tEBN##################
class TemporalEffectiveBatchNormNd(base.MemoryModule):
    bn_instance = _BatchNorm
    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        super().__init__()

        self.bn = self.bn_instance(num_features, eps, momentum, affine, track_running_stats, step_mode)
        self.scale = nn.Parameter(torch.ones([T]))
        self.register_memory('t', 0)

    def single_step_forward(self, x: torch.Tensor):
        return self.bn(x) * self.scale[self.t]

class TemporalEffectiveBatchNorm2d(TemporalEffectiveBatchNormNd):
    bn_instance = layer.BatchNorm2d   
    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm2d-en>`

        .. _TemporalEffectiveBatchNorm2d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm2d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm2d-cn>`

        .. _TemporalEffectiveBatchNorm2d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm2d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, H, W]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1)
    
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
    

class SEW_Resblock_con5_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T, connection):
        super(SEW_Resblock_con5_with_BNTT_and_mines, self).__init__()
        self.connection = connection
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True)
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)
        if self.connection == 'add':
            out = out + x_identity
        elif self.connection == 'and':
            out = out * x_identity
        elif self.connection == 'iand':
            out = out * (1 - x_identity)
        return out, out_l1_penalty_1

class SEW_Resblock_con3_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T, connection):
        super(SEW_Resblock_con3_with_BNTT_and_mines, self).__init__()
        self.T = T
        self.connection = connection
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True)
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lif2(out)
        if self.connection == 'add':
            out = out + x_identity
        elif self.connection == 'and':
            out = out * x_identity
        elif self.connection == 'iand':
            out = out * (1 - x_identity)
        return out, out_l1_penalty_1

class Inception_like_FusionNet_with_SEW_BNTT_and_mines(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, T:int, connect):
        super(Inception_like_FusionNet_with_SEW_BNTT_and_mines, self).__init__()
        self.T = T
        self.connect = connect
        self.conv1 = layer.Conv2d(in_ch, mid_ch, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv_3 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=mid_ch)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.resblock1 = SEW_Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock2 = SEW_Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock3 = SEW_Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock4 = SEW_Resblock_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)

        self.resblock_1 = SEW_Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_2 = SEW_Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_3 = SEW_Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_4 = SEW_Resblock_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)

        self.conv2 = layer.Conv2d(mid_ch * 2, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = layer.Conv2d(mid_ch * 2, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv_31 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = layer.BatchNorm2d(out_ch)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_ch)
        self.bn3 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_ch)
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
        x, x_l1_penalty_res_1 = self.resblock1(x)
        x, x_l1_penalty_res_2 = self.resblock2(x)
        x, x_l1_penalty_res_3 = self.resblock3(x)
        x, x_l1_penalty_res_4 = self.resblock4(x)

        x1, x_l1_penalty_res_5 = self.resblock_1(x)
        x1, x_l1_penalty_res_6 = self.resblock_2(x1)
        x1, x_l1_penalty_res_7 = self.resblock_3(x1)
        x1, x_l1_penalty_res_8 = self.resblock_4(x1)

        x = torch.cat([x, x1], dim=2)
        # print(x.shape)
        
        x_mines = self.conv3(x)
        x_mines = self.bn3(x_mines)
        x_mines = self.lif3(x_mines)
        x_mines_l1_penalty_1 = torch.norm(x_mines, 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x_l1_penalty_2 = torch.norm(x, 1)
        
        result_mines = self.lif3.v_seq.permute(1, 2, 3, 4, 0)
        result = self.lif2.v_seq.permute(1, 2, 3, 4, 0) # [N,C,H,W,T]
        result = torch.cat([result, -result_mines], dim=-1)# [N,C,H,W,2T]
        result = torch.mean(result, dim=-1)
        output = torch.tanh(result)
        l1_penalty = x_l1_penalty_1 + x_l1_penalty_2 + \
                     x_l1_penalty_res_1 + x_l1_penalty_res_2 + x_l1_penalty_res_3 + x_l1_penalty_res_4 + \
                     x_l1_penalty_res_5 + x_l1_penalty_res_6 + x_l1_penalty_res_7 + x_l1_penalty_res_8 + \
                     x_mines_l1_penalty_1
        return output, l1_penalty
    
