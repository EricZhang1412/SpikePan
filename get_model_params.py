# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from spike_fusionnet_model import FusionNet, FusionNet_8_res, Inception_like_FusionNet_with_SEW_BNTT_and_mines, Inception_like_FusionNet_without_bn
from ptflops import get_model_complexity_info
from spikingjelly.activation_based import monitor, neuron, functional, layer

import sys
sys.stdout = open('model_params.txt', 'w')


T = 16
N = 1
# Model
print('==> Building model..')
model = Inception_like_FusionNet_with_SEW_BNTT_and_mines(8, 32, 8, T=T, connect='add')
# cnt = 0
# for module in model.modules():
#     for param in module.parameters():
#         # print module TYPE and parameter size  
#         print(module, param.size())
        
#         cnt += param.numel()
# print('Total number of parameters: %d' % cnt)



spike_seq_monitor = monitor.OutputMonitor(model, neuron.LIFNode)
for param in model.parameters():
    param.data.abs_()

x_seq = torch.rand([N, 8, 64, 64])
with torch.no_grad():
    model(x_seq)

print(f'spike_seq_monitor.records=\n{spike_seq_monitor.records}')
print(f'model={model}')
print(f'spike_seq_monitor.monitored_layers={spike_seq_monitor.monitored_layers}')

spike_seq_monitor.remove_hooks()

sys.stdout.close()