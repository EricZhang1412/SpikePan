import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


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
    
# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self,T:int):
        super(Resblock, self).__init__()
        self.T = T
        channel = 32
        self.conv1 = layer.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=channel)
        self.conv2 = layer.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=channel)
        # self.relu = nn.ReLU(inplace=True)
        self.neuron1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True,surrogate_function=surrogate.ATan())
        self.neuron2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True,surrogate_function=surrogate.ATan())
    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.neuron1(self.conv1(x))  # Bsx32x64x64
        rs1 = self.bn1(rs1)
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs1 = self.bn2(rs1)
        rs = torch.add(x, rs1)  # Bsx32x64x64
        rs = self.neuron2(rs)
        return rs

# -----------------------------------------------------
class PanNet(nn.Module):
    def __init__(self, T: int):
        super(PanNet, self).__init__()
        self.T = T
        channel = 32
        spectral_num = 8

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = layer.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        # self.deconv1 = layer.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8,
        #                                     stride=4,
        #                                     padding=2, bias=True)
        #8*h*w=>8*4h*4w

        self.conv1 = layer.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn1 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=channel)
        self.neur1 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.res1 = Resblock(T=self.T)
        self.res2 = Resblock(T=self.T)
        self.res3 = Resblock(T=self.T)
        self.res4 = Resblock(T=self.T)
        self.conv2 = layer.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = layer.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.bn2 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=channel)
        self.bn3 = tdBatchNorm(alpha=1.0, v_th=1.0, num_features=channel)
        # self.relu = nn.ReLU(inplace=True)
        self.neur2 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.neur3 = neuron.LIFNode(tau=2.0, detach_reset=True, store_v_seq=True, surrogate_function=surrogate.ATan())
        self.backbone = nn.Sequential(# 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        functional.set_step_mode(self, step_mode='m')
        # if use_cupy:
        #     functional.set_backend(self, backend='cupy')
        init_weights(self.backbone, self.deconv, self.conv1, self.conv3)   # state initialization, important!

    def forward(self, x, y):  # x= hp of ms; # Bsx8x16x16 y = hp of pan # Bsx1x64x64
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # pixel shuffle
        output_deconv = self.deconv(x)  # Bsx8x64x64
        # output_deconv = output_deconv.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

        y = y.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        input = torch.cat([output_deconv, y], dim=2)  # Bsx9x64x64
        # input = self.neur1(input)
        rs = self.neur1(self.conv1(input))  # Bsx32x64x64
        rs = self.bn1(rs)
        rs = self.backbone(rs)  # ResNet's backbone! # Bsx32x64x64

        x_mines = self.conv3(rs)  # Bsx8x64x64
        x_mines = self.bn3(x_mines)
        x_mines = self.neur3(x_mines)
        x = self.conv2(rs)
        x = self.bn2(x)
        x = self.neur2(x)
        result_mines = self.neur3.v_seq.permute(1, 2, 3, 4, 0)
        result = self.neur2.v_seq.permute(1, 2, 3, 4, 0) # [N,C,H,W,T]
        result = torch.cat([result, -result_mines], dim=-1)# [N,C,H,W,2T]
        result = torch.mean(result, dim=-1)
        output = torch.tanh(result)
        # print(output.shape)
        return output

# ----------------- End-Main-Part ------------------------------------

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
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
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
    # if grad:
    #     from torchsummary import summary
    #     summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    #     # summary(model, input_size=[(4, 8, 16, 16), (4, 1, 64, 64)], batch_size=1)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    #
    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model,(x,))



def inspect_weight_decay():
    ...


# device = torch.device('cuda')  # 指定使用 CUDA 设备
#
#   # 将输入张量移动到 GPU
#
# # 此处进行模型的前向计算
# model = PanNet(T=4).cuda()
# x_size = (32, 8, 16, 16)
# y_size = (32, 1, 64, 64)
#
# x = torch.randn(*x_size)
# y = torch.randn(*y_size)
# x = x.to(device)
# y = y.to(device)
# # model.train()
# hp_sr = model(x, y)
# # print(hp_sr.shape)