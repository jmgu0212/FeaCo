# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn
import math
from collections import OrderedDict

class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()
        # self.conv1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1, groups=1)
        self.conv1 = nn.Sequential(
        OrderedDict(
            [
                ('conv', nn.Conv3d(2, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1)),
                ('activation', nn.ReLU()),
            ]
        )
    )
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
    def forward(self, x, record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []
        
        for xx in split_x:
            xx_max = torch.max(xx, dim=0, keepdim=True)[0]
            xx_avg = torch.mean(xx, dim=0, keepdim=True)
            F_Sp = torch.cat((xx_max,xx_avg),dim = 0).unsqueeze(0)
            # F_Sp = F_Sp.permute(0,2,1,3,4)
            # print(F_Sp.shape)
            # exit()
            # conv = nn.Conv3d(2, 1, kernel_size=(3,3,3), stride=1, padding=1)
            xx = self.conv1(F_Sp)[0]
            # print(aa.shape)
            # exit()
            out.append(xx)
        return torch.cat(out, dim=0)