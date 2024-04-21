# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from numpy import record
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from v2xvit.models.sub_modules.pillar_vfe import PillarVFE
from v2xvit.models.sub_modules.point_pillar_scatter import PointPillarScatter
from v2xvit.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from v2xvit.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from v2xvit.models.sub_modules.downsample_conv import DownsampleConv
from v2xvit.models.sub_modules.naive_compress import NaiveCompressor
from v2xvit.models.fuse_modules.where2comm_mutihead import Where2comm
from v2xvit.models.sub_modules.psm_mask import Communication
from v2xvit.models.sub_modules.positioning_error_correction import get_t_matrix
import torch
    

def warp_affine_simple(src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):

    B, C, H, W = src.size()
    grid = F.affine_grid(M,
                         [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

def trans_tx(t_matrix,H,W):
    a00 = t_matrix[0][0]
    a01 = t_matrix[0][1]
    a02 = t_matrix[0][2]
    a12 = t_matrix[1][2]
    tx = (a02 - ((1 - a00) * W * 0.5 - a01 * H * 0.5))/(W * 0.5)
    ty = (a12 - (a01 * W * 0.5 + (1 - a00) * H * 0.5))/(H * 0.5)
    
    T = np.float32([[a00,a01,tx],
                    [a01,a00,ty],
                    [0,0,1]])
                    
    hang = np.linalg.det(T)
    if hang == 0:
        return  np.float32([[1,0,0],
                    [0,1,0]])
    a = np.linalg.inv(T)
    return a[0:2,:]

class PointPillarWhere2comm(nn.Module):
    def __init__(self, args):
        super(PointPillarWhere2comm, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # if 'resnet' in args['base_bev_backbone'] and args['base_bev_backbone']['resnet']:
        #     self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.dcn = False

        # self.fusion_net = TransformerFusion(args['fusion_args'])
        self.fusion_net = Where2comm(args['fusion_args'])
        self.multi_scale = args['fusion_args']['multi_scale']

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        
        self.naive_communication = Communication(args['communication'])

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        
        record_len = data_dict['record_len']

        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c 
        batch_dict = self.pillar_vfe(batch_dict)  
        # n, c -> N, C, H, W 
        batch_dict = self.scatter(batch_dict)
        # N, C, H', W'  
        batch_dict = self.backbone(batch_dict) 
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        if self.shrink_flag: 
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        
        psm_single = self.cls_head(spatial_features_2d)

        split_psm_single = self.regroup(psm_single, record_len)

        _, communication_masks = self.naive_communication(split_psm_single, record_len)
        
        split_spatial_features_2d = self.regroup(batch_dict['spatial_features'], record_len) 
        feature_list = []

        for i in range(len(communication_masks)):
            mask = communication_masks[i].squeeze(1).to('cpu').numpy()
            cav_num,mask_h,mask_w = mask.shape
            ego_mask = mask[0]
            feature_list.append(split_spatial_features_2d[i][0].unsqueeze(0))

            for j in range(1,cav_num):
                features_2d = split_spatial_features_2d[i][j]
                C,H,W = features_2d.shape
                other_mask = mask[j]

                t_matrix = get_t_matrix(ego_mask,other_mask) # get fine-grid transformation matrix
                t_matrix = trans_tx(t_matrix,mask_h,mask_w)
                t_matrix = torch.from_numpy(t_matrix).to('cuda').unsqueeze(0)
                features_2d = warp_affine_simple(features_2d.unsqueeze(0),t_matrix,(H,W))
                feature_list.append(features_2d)


        batch_dict['spatial_features'] = torch.vstack(feature_list)
        
        if self.multi_scale:
            fused_feature= self.fusion_net(batch_dict['spatial_features'],
                                            record_len,
                                            self.backbone)
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
            
        else:
            # batch_dict = self.backbone(batch_dict) 
            # spatial_features_2d = batch_dict['spatial_features_2d']
            # if self.shrink_flag:  
            #     spatial_features_2d = self.shrink_conv(spatial_features_2d)
            
            fused_feature = self.fusion_net(spatial_features_2d,
                                            record_len,)
            
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)


        output_dict = {'psm': psm,
                       'rm': rm,
                       }
        

        return output_dict
