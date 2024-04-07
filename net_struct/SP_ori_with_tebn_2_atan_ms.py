import torch
import torch.nn as nn
import numpy as np
import math
from spikingjelly.activation_based import base, neuron, encoding, functional, surrogate, layer

import torch.nn.init as int
import h5py
import torch.utils.data as data
from torch.nn.modules.batchnorm import _BatchNorm

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

class MS_Resblock_4(nn.Module):
    def __init__(self, in_channels, out_channels, T:int):
        super(MS_Resblock_4, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.mid_lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.conv3 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv4 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif4 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.mid_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.conv5 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif5 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv6 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif6 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.mid_lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())

        self.conv7 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif7 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv8 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        self.lif8 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())



    def forward(self, x):
        x_identity = x
        out = self.lif1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.lif2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x_identity
        
        out_2 = self.mid_lif1(out)
        out_2_identity = self.mid_lif1.v_seq
        out_2 = self.conv3(out_2)
        out_2 = self.bn3(out_2)
        out_2 = self.lif3(out_2)
        out_2 = self.conv4(out_2)
        out_2 = self.bn4(out_2)
        out_2 += out_2_identity

        out_3 = self.mid_lif2(out_2)
        out_3_identity = self.mid_lif2.v_seq
        out_3 = self.conv5(out_3)
        out_3 = self.bn5(out_3)
        out_3 = self.lif5(out_3)
        out_3 = self.conv6(out_3)
        out_3 = self.bn6(out_3)
        out_3 += out_3_identity
        
        out_4 = self.mid_lif3(out_3)
        out_4_identity = self.mid_lif3.v_seq
        out_4 = self.conv7(out_4)
        out_4 = self.bn7(out_4)
        out_4 = self.lif7(out_4)
        out_4 = self.conv8(out_4)
        out_4 = self.bn8(out_4)
        out_4 += out_4_identity
        return out_4

    
class FusionNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, T:int):
        super(FusionNet, self).__init__()
        self.T = T
        self.conv1 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=mid_ch)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        # MS_Resblock_4
        self.resblock = MS_Resblock_4(mid_ch, mid_ch, T=self.T)
        self.conv2 = layer.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_ch)
        self.bn3 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_ch)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.conv3 = layer.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.resblock(x)
        
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
        return output
