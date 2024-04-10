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
from ANN_fusionnet_model import FusionNet
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


model = FusionNet(8, 32, 8)
model_input_1 = torch.randn(1, 8, 256, 256)

y = model(model_input_1)
print(summary(model, model_input_1, show_input=False, show_hierarchical=False))