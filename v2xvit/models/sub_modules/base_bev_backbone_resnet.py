import numpy as np
import torch
import torch.nn as nn

# from v2xvit.models.sub_modules.resblock import ResNetModified, BasicBlock,Bottleneck,Bottle2neck
from v2xvit.models.sub_modules.res2net import Res2Net,Bottle2neck
DEBUG = False

class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out


class ResNetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        # self.resnet = ResNetModified(BasicBlock, 
        #                                 layer_nums,
        #                                 layer_strides,
        #                                 num_filters)
        
        self.resnet = Res2Net(Bottle2neck, layer_nums, baseWidth = 26, scale = 4)

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.layer_fusion = LAM_Module_v2(c_in)

    def forward(self, data_dict):
        
        #[2,64,200,704]
        spatial_features = data_dict['spatial_features']
        
       #[2,64,100,352],[2,128,50,176],[2,256,25,88]
        x = self.resnet(spatial_features)  # tuple of features
        
        ups = []

        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            # x = torch.cat(ups, dim=1)
            inp_fusion= torch.cat([ups[0].unsqueeze(1), ups[1].unsqueeze(1), ups[2].unsqueeze(1)], dim=1)
            x = self.layer_fusion(inp_fusion)

        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict
