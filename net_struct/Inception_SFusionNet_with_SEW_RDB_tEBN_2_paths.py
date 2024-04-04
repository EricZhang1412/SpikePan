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
    
    
class SEW_RDB_con5_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T, connection):
        super(SEW_RDB_con5_with_BNTT_and_mines, self).__init__()
        self.connection = connection
        self.T = T
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True)
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.lif2(out)
        # out_l1_penalty_2 = torch.norm(out, 1)
        # if self.connection == 'add':
        #     out = out + x_identity
        # elif self.connection == 'and':
        #     out = out * x_identity
        # elif self.connection == 'iand':
        #     out = out * (1 - x_identity)
        return x_identity, out, out_l1_penalty_1
        # return x_identity, out, out_l1_penalty_1, out_l1_penalty_2
    
class SEW_RDB_con3_with_BNTT_and_mines(nn.Module):
    def __init__(self, in_channels, out_channels, T, connection):
        super(SEW_RDB_con3_with_BNTT_and_mines, self).__init__()
        self.T = T
        self.connection = connection
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=out_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True)
        # functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x_identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lif1(out)
        out_l1_penalty_1 = torch.norm(out, 1)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.lif2(out)
        # out_l1_penalty_2 = torch.norm(out, 1)
        # if self.connection == 'add':
        #     out = out + x_identity
        # elif self.connection == 'and':
        #     out = out * x_identity
        # elif self.connection == 'iand':
        #     out = out * (1 - x_identity)
        return x_identity, out, out_l1_penalty_1
        # return x_identity, out, out_l1_penalty_1, out_l1_penalty_2

class Inception_like_FusionNet_with_SEW_RDB_BNTT_and_mines(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, T:int, connect):
        super(Inception_like_FusionNet_with_SEW_RDB_BNTT_and_mines, self).__init__()
        self.T = T
        self.connect = connect
        self.conv1 = layer.Conv2d(in_ch, mid_ch, kernel_size=5, stride=1, padding=2, bias=False)
        # self.conv_3 = layer.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TemporalEffectiveBatchNorm2d(T=self.T, num_features=mid_ch)
        # self.relu1 = nn.ReLU(inplace=True)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.resblock1 = SEW_RDB_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock2 = SEW_RDB_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock3 = SEW_RDB_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock4 = SEW_RDB_con5_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)

        self.resblock_1 = SEW_RDB_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_2 = SEW_RDB_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_3 = SEW_RDB_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)
        self.resblock_4 = SEW_RDB_con3_with_BNTT_and_mines(mid_ch, mid_ch, T=self.T, connection=self.connect)

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
        x1_indentity, x1_out, x1_l1_penalty_1 = self.resblock1(x)
        x2_indentity, x2_out, x2_l1_penalty_1 = self.resblock2(x1_out)
        x3_indentity, x3_out, x3_l1_penalty_1 = self.resblock3(x2_out)
        x4_indentity, x4_out, x4_l1_penalty_1 = self.resblock4(x3_out)

        if (self.connect == 'add'):
            x_res1_out = x1_indentity + x2_indentity + x3_indentity + x4_indentity + x4_out
        elif (self.connect == 'and'):
            x_res1_out = x1_indentity * x2_indentity * x3_indentity * x4_indentity * x4_out
        elif (self.connect == 'iand'):
            x_res1_out = (1 - x1_indentity) * (1 - x2_indentity) * (1 - x3_indentity) * (1 - x4_indentity) * x4_out # 合理与否有待考证

        x1_identity_2, x1_out_2, x1_l1_penalty_1_1 = self.resblock_1(x)
        x2_indentity_2, x2_out_2, x2_l1_penalty_1_1 = self.resblock_2(x1_out_2)
        x3_indentity_2, x3_out_2, x3_l1_penalty_1_1 = self.resblock_3(x2_out_2)
        x4_indentity_2, x4_out_2, x4_l1_penalty_1_1 = self.resblock_4(x3_out_2)

        if (self.connect == 'add'):
            x_res2_out = x1_identity_2 + x2_indentity_2 + x3_indentity_2 + x4_indentity_2 + x4_out_2
        elif (self.connect == 'and'):
            x_res2_out = x1_identity_2 * x2_indentity_2 * x3_indentity_2 * x4_indentity_2 * x4_out_2
        elif (self.connect == 'iand'):
            x_res2_out = (1 - x1_identity_2) * (1 - x2_indentity_2) * (1 - x3_indentity_2) * (1 - x4_indentity_2) * x4_out_2 # 合理与否有待考证

        x = torch.cat([x_res1_out, x_res2_out], dim=2)
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
                     x1_l1_penalty_1 + x2_l1_penalty_1 + \
                     x3_l1_penalty_1 + x4_l1_penalty_1 + \
                     x1_l1_penalty_1_1 + x2_l1_penalty_1_1 + \
                     x3_l1_penalty_1_1 + x4_l1_penalty_1_1 + \
                     x_mines_l1_penalty_1
        return output, l1_penalty


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, layer.Conv2d):   ## initialization for Conv2d

                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, layer.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, layer.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False):
    pass
    # if grad:
    #     from torchsummary import summary
    #     summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    #
    # if writer is not None:
    #     x = torch.randn(1, 64, 64, 64)
    #     writer.add_graph(model,(x,))



def inspect_weight_decay():
    ...