# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of V2VNet Fusion
"""

from ast import For
from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from math import sqrt
from torch.nn import init

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # print(channels_per_group)
    # reshape
    # b, c, h, w =======>  b, g, c_per, h, w
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x

class PSA(nn.Module):
    def __init__(self, channel=512,reduction=8,S=5):
        super(PSA, self).__init__()
        self.S=S

        self.convs =  nn.ModuleList()
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))

        self.se_blocks = nn.ModuleList()
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)
        
        self.conv1 = nn.Sequential(OrderedDict(
            [
                ('conv', nn.Conv3d(2, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1)),
                ('activation', nn.ReLU(inplace=True)),
            ]
        ))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        
        #Step1:SPC module
        spc_out = []
        SPC_out=x.view(b,self.S,c//self.S,h,w) #bs,s,ci,h,w
        for i in range(self.S):
           spc_out.append(self.convs[i](SPC_out[:,i,:,:,:]))
        SPC_outs = torch.stack(spc_out,dim=1)
        #Step2:SE weight
        se_out=[]
        for i in range(self.S):
            se_out.append(self.se_blocks[i](SPC_outs[:,i,:,:,:]))
        SE_out=torch.stack(se_out,dim=1)
        SE_out=SE_out.expand_as(SPC_out)

        #Step3:Softmax
        softmax_out=self.softmax(SE_out)

        #Step4:SPA
        PSA_out=SPC_out*softmax_out
        PSA_out_max = torch.max(PSA_out,dim=1)[0]
        PSA_out_mean = torch.mean(PSA_out,dim=1)
        PSA_out_max=PSA_out_max.view(b,-1,h,w)
        PSA_out_mean = PSA_out_mean.view(b,-1,h,w)
        PSA_out = torch.cat((PSA_out_max,PSA_out_mean),dim=0).unsqueeze(0)
        PSA_out = self.conv1(PSA_out).squeeze(0)
        
        return PSA_out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class MS_CAM(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(MS_CAM, self).__init__()
        mid_channel = channel // ratio
        self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channel),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        g_x = self.global_att(x)
        l_x = self.local_att(x)
        w = self.sigmoid(l_x * g_x.expand_as(l_x))
        return w * x

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=5):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) #bs,k,n,c
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out

class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model,1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input) #(bs,nq,1)
        weight_i = torch.softmax(i, dim=1) #bs,nq,1
        context_score = weight_i * self.fc_k(input) #bs,nq,d_model
        context_vector = torch.sum(context_score,dim=1,keepdim=True) #bs,1,d_model
        v = self.fc_v(input) * context_vector #bs,nq,d_model
        out = self.fc_o(v) #bs,nq,d_model

        return out

class Involution(nn.Module):
    def __init__(self, kernel_size, in_channel=4, stride=1, group=1,ratio=4):
        super().__init__()
        self.kernel_size=kernel_size
        self.in_channel=in_channel
        self.stride=stride
        self.group=group
        assert self.in_channel%group==0
        self.group_channel=self.in_channel//group
        self.conv1=nn.Conv2d(
            self.in_channel,
            self.in_channel//ratio,
            kernel_size=1
        )
        self.bn=nn.BatchNorm2d(in_channel//ratio)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(
            self.in_channel//ratio,
            self.group*self.kernel_size*self.kernel_size,
            kernel_size=1
        )
        self.avgpool=nn.AvgPool2d(stride,stride) if stride>1 else nn.Identity()
        self.unfold=nn.Unfold(kernel_size=kernel_size,stride=stride,padding=kernel_size//2)
        
    
    def forward(self, inputs):
        B,C,H,W=inputs.shape
        weight=self.conv2(self.relu(self.bn(self.conv1(self.avgpool(inputs))))) #(bs,G*K*K,H//stride,W//stride)
        b,c,h,w=weight.shape
        weight=weight.reshape(b,self.group,self.kernel_size*self.kernel_size,h,w).unsqueeze(2) #(bs,G,1,K*K,H//stride,W//stride)

        x_unfold=self.unfold(inputs)
        x_unfold=x_unfold.reshape(B,self.group,C//self.group,self.kernel_size*self.kernel_size,H//self.stride,W//self.stride) #(bs,G,G//C,K*K,H//stride,W//stride)

        out=(x_unfold*weight).sum(dim=3)#(bs,G,G//C,1,H//stride,W//stride)
        out=out.reshape(B,C,H//self.stride,W//self.stride) #(bs,C,H//stride,W//stride)

        return out

class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)
    
class DynamicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0,dilation=1,grounps=1,bias=True,K=8,temprature=30,ratio=4,init_weight=True):
        super().__init__()
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.init_weight=init_weight
        self.attention=Attention(in_planes=in_planes,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)
        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(x) #bs,K
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        
        output=output.view(bs,self.out_planes,h,w)
        return output
    
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b,c,h,w = x.shape
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        index = torch.topk(y, k=int(c/5), dim=2, largest=True)[1]
        index = index.reshape(-1)
        # print(index)
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        out =  x * y.expand_as(x)
        out = out.index_select(1, index)
        return out

from torch.nn import functional as F

class OutlookAttention(nn.Module):

    def __init__(self,dim,num_heads=1,kernel_size=3,padding=1,stride=1,qkv_bias=False,
                    attn_drop=0.1):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.scale=self.head_dim**(-0.5)

        self.v_pj=nn.Linear(dim,dim,bias=qkv_bias)
        self.attn=nn.Linear(dim,kernel_size**4*num_heads)

        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(attn_drop)

        self.unflod=nn.Unfold(kernel_size,padding,stride) #手动卷积
        self.pool=nn.AvgPool2d(kernel_size=stride,stride=stride,ceil_mode=True) 

    def forward(self, x) :
        B,H,W,C=x.shape

        #映射到新的特征v
        v=self.v_pj(x).permute(0,3,1,2) #B,C,H,W
        h,w=math.ceil(H/self.stride),math.ceil(W/self.stride)
        v=self.unflod(v).reshape(B,self.num_heads,self.head_dim,self.kernel_size*self.kernel_size,h*w).permute(0,1,4,3,2) #B,num_head,H*W,kxk,head_dim

        #生成Attention Map
        attn = x[0].unsqueeze(0).repeat(B,1,1,1)
        attn=self.pool(attn.permute(0,3,1,2)).permute(0,2,3,1) #B,H,W,C
        attn=self.attn(attn).reshape(B,h*w,self.num_heads,self.kernel_size*self.kernel_size \
                    ,self.kernel_size*self.kernel_size).permute(0,2,1,3,4) #B，num_head，H*W,kxk,kxk
        attn=self.scale*attn
        attn=attn.softmax(-1)
        attn=self.attn_drop(attn)

        #获取weighted特征
        out=(attn @ v).permute(0,1,4,3,2).reshape(B,C*self.kernel_size*self.kernel_size,h*w) #B,dimxkxk,H*W
        out=F.fold(out,output_size=(H,W),kernel_size=self.kernel_size,
                    padding=self.padding,stride=self.stride) #B,C,H,W
        out=self.proj(out.permute(0,2,3,1)) #B,H,W,C
        out=self.proj_drop(out)

        return out
    
class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()

        self.PSA = PSA(feature_dim*5,reduction=4,S=5)
        self.SA = SplitAttention(feature_dim)
        self.MVA = MobileViTv2Attention(feature_dim)

        self.involution1=Involution(kernel_size=5,in_channel=feature_dim*5,stride=1)
        self.att = ScaledDotProductAttention(feature_dim)
        self.ecaattention = ECAAttention(kernel_size=3)

        self.DynamicConvs = nn.Sequential(
            DynamicConv(in_planes=feature_dim*5,out_planes=feature_dim//2,kernel_size=3,stride=1,padding=1,bias=False),
            DynamicConv(in_planes=feature_dim//2,out_planes=feature_dim//2//2,kernel_size=3,stride=1,padding=1,bias=False),
            DynamicConv(in_planes=feature_dim//2//2,out_planes=feature_dim,kernel_size=3,stride=1,padding=1,bias=False)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(5, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1),
            # nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(5, 1, kernel_size=(5,5,5),stride=1, padding=2, groups=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True),
        )
        self.br = nn.Sequential(
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.outlook = OutlookAttention(dim=feature_dim)
        self.danet=DAModule(feature_dim)

    def forward(self, x):

        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)

        # # x = x.view(1,5*C, H,W).to(x.device)
        # # x = channel_shuffle(x,5)
        # # x = self.PSA(x)[0]
        # x = x.unsqueeze(0).permute(0,1,3,4,2)
        # x = self.SA(x).permute(0,3,1,2)[0]
        

        # cav_num, C, H, W = x.shape
        # x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        # x = self.MVA(x)
        # x = x.permute(1, 2, 0).view(cav_num, C, H, W)  # C, W, H before
        


        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)
        # x = torch.split(x, 1, dim=0)
        # x = torch.cat(x, dim = 1)
        # x = channel_shuffle(x,5)
        # # print(x.shape)
        # x=self.involution(x)
        # x = channel_shuffle(x,C)
        # x = torch.split(x, C, dim=1)
        # x = torch.vstack(x)
        # x = torch.max(x, dim=0, keepdim=False)[0]

        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)
        # x = torch.split(x, 1, dim=0)
        # x = torch.cat(x, dim = 1)
        # x = channel_shuffle(x,5)
        # # print(x.shape)
        # x=self.ecaattention(x)[0]
        
        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)
        # x = torch.split(x, 1, dim=0)
        # x = torch.cat(x, dim = 1)
        # x = channel_shuffle(x,5)
        # # print(x.shape)
        # x = self.DynamicConvs(x)
        # x = channel_shuffle(x,C)[0]

        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)
        # x_shuffle = torch.split(x, 1, dim=0)
        # x_shuffle = torch.cat(x_shuffle, dim = 1)
        # x_shuffle = channel_shuffle(x_shuffle,5)
        # x_shuffle = torch.split(x_shuffle, C, dim=1)
        # x_shuffle = torch.vstack(x_shuffle)
        # x_shuffle  = x_shuffle.unsqueeze(0)
        # x_shuffle = self.conv1(x_shuffle)
        # x = x.unsqueeze(0)
        # x = self.conv2(x)
        # xx = torch.cat((x,x_shuffle), dim = 1)
        # x = self.conv3(xx)[0][0]
        # print(x.shape)

        # cav_num, C, H, W = x.shape
        # x = x.permute(0,2,3,1)
        # x = self.outlook(x)
        # x = x.permute(0,3,1,2)
        # x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        # x = self.att(x, x, x)
        # x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before

        # cav_num, C, H, W = x.shape
        # x = x.view(cav_num, C, -1).permute(2, 0, 1) #(H*W, cav_num, C), perform self attention on each pixel.
        # x = self.att(x, x, x)
        # x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before

        # cav_num, C, H, W = x.shape
        # if cav_num < 5:
        #     padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
        #     x = torch.cat((x,padding),dim=0)
        # # x_ego = x[0]
        # x = x.view(5, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        # x = self.att(x, x, x)
        # x = x.permute(1, 2, 0).view(5,C, H, W).unsqueeze(0)  # C, W, H before
        # x = self.conv1(x)[0]
        # x = self.br(x)[0]

        x = self.danet(x)[0]
        
        return x
    
class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.att  = MultiHeadSelfAttention(channels,channels,channels,8)
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.attn_fusion = AttenFusion(channels)
        self.mscam = MS_CAM(channels)

        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, x):
        # if x.shape[0]>1:
        #     x = self.mscam(x)
        cav_num, C, H, W = x.shape
        # x = add_pe_map(x)
        # x = add_pe_map(x)
        x = x.view(cav_num, C, -1)  #(H*W,cav_num,C)
        query = x.permute(2, 0, 1)
        key = x.permute(2, 0, 1)
        value = x.permute(2, 0, 1)
        fused_feature = self.att(query)   # C, W, H before
        # fused_feature = self.attn_fusion(x)
        fused_feature = fused_feature.permute(1, 2, 0).view(cav_num, C, H, W)[0]
        # print("a")
        return fused_feature

def add_pe_map(x):
    # scale = 2 * math.pi
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

    if len(x.shape) == 4:
        x_pe = x + pos[None,:,:,:]
    elif len(x.shape) == 5:
        x_pe = x + pos[None,None,:,:,:]
    return x_pe

class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()
        # self.conv1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3,3,3), stride=1, padding=1, groups=1)        

        self.conv2 = nn.Sequential(OrderedDict(
            [
                ('conv', nn.Conv3d(5, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1)),
                ('activation', nn.ReLU(inplace=True)),
            ]
        ))

        self.conv3 = nn.Sequential(
           nn.Conv3d(5, 1, kernel_size=(5,5,5),stride=1, padding=2, groups=1),
           nn.ReLU(inplace=True),
           )
        
    
    def forward(self, xx):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        
        xx_stack = []
    
        xx_ego = xx[0].unsqueeze(0)
        xx_max = torch.max(xx, dim=0, keepdim=True)[0] #遮挡特征+重合特征
        xx_avg = torch.mean(xx, dim=0, keepdim=True)  #稀疏特征
        xx_min = torch.min(xx,dim=0,keepdim=True)[0]  #重合特征
        xx_max_min = xx_max - xx_min

        # xx_stack.append(xx_ego)
        xx_stack.append(xx_max)
        xx_stack.append(xx_avg)
        xx_stack.append(xx_max_min)
        
        cav_num, C, H, W = xx.shape
        if cav_num < 5:
           padding =  torch.zeros([5-cav_num,C,H,W]).to(xx.device)
           xx = torch.cat((xx,padding),dim=0)

        xx = xx.unsqueeze(0)
        s_1 = self.conv2(xx)[0]
        s_2 = self.conv3(xx)[0]

        xx_stack.append(s_1)
        xx_stack.append(s_2)

        F_Sp = torch.vstack(xx_stack).unsqueeze(0)
        # F_Sp = torch.cat((F_Sp,xx),dim = 0).unsqueeze(0)

        out = self.conv1(F_Sp).squeeze(0)[0]
        # print(out.shape)
        # exit()
        return out

class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self,d_model,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model

        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout=nn.Dropout(dropout)

        # self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        
        #C,cav,h*w
        b_s,c,d_model = queries.shape 
        keys = keys.transpose(1, 2)
        att = torch.matmul(queries, keys) / np.sqrt(d_model)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        # att=self.dropout(att)

        out = torch.matmul(att, values)  # (b_s, nq, h*d_v)
        # out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model):
        super().__init__()
        # self.cnn=nn.Conv2d(d_model,d_model,kernel_size=3,padding=(3-1)//2)
        self.pa=ScaledDotProductAttention(d_model)
    
    def forward(self,x):
        cav_num,c,h,w=x.shape
        # y=self.cnn(x)
        # y = add_pe_map(x)
        y =x
        y=y.view(cav_num,c,-1).permute(2,0,1) #h*w,cav_num,c
        y=self.pa(y,y,y) 
        y=y.permute(1,2,0).view(cav_num,c,h,w)
        return y

class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model):
        super().__init__()
        # self.cnn=nn.Conv2d(d_model,d_model,kernel_size=3,padding=(3-1)//2)
        # self.pa =SimplifiedScaledDotProductAttention(d_model)
        self.pa=ScaledDotProductAttention(d_model)
    
    def forward(self,x):
        cav_num,c,h,w=x.shape
        # y=self.cnn(x)
        y = x
        # y = y.view(cav_num,c,-1) #Ca,C,h*w
        y = y.view(cav_num,c,-1).permute(1,0,2) #C,CAV_NUM,h*w
        y=self.pa(y,y,y) 
        y=y.permute(1,0,2).view(cav_num,c,h,w)
        # y=y.view(cav_num,c,h,w)
        return y

class DAModule(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=C)
        self.channel_attention_module=ChannelAttentionModule(d_model=C)
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3,stride=1, padding=1, groups=1),
            nn.Sigmoid()
        )
        # self.a1 = torch.nn.Parameter(torch.tensor([0.34]), requires_grad=True)
        # self.a2 = torch.nn.Parameter(torch.tensor([0.33]), requires_grad=True)
        # self.a3 = torch.nn.Parameter(torch.tensor([0.33]), requires_grad=True)
        self.fuse_conv = nn.Sequential(
           nn.Conv3d(2, 1, kernel_size=(3,3,3),stride=1, padding=1, groups=1),
           nn.ReLU(inplace=True),
           )
    def forward(self,input):
        cav_num,c,h,w=input.shape
        # out = []
        # for i in range(cav_num):
        #     cav_i = input[i].unsqueeze(0).repeat(cav_num,1,1,1)
        #     cha = input-cav_i
        #     cha[i] = input[i]
        #     cha = self.conv1(cha)
        #     cha = self.position_attention_module(cha)
        #     # cha = torch.mean(cha, dim=0, keepdim=True) + torch.max(cha, dim=0, keepdim=True)[0]
        #     out.append(cha[i].unsqueeze(0))       
        # out = torch.vstack(out)

        p_out=self.position_attention_module(input)
        # c_out=self.channel_attention_module(input)
        # res =torch.cat((p_out[0].unsqueeze(0),c_out[0].unsqueeze(0)),dim=0)
        # res =torch.cat((p_out[0].unsqueeze(0),torch.max(c_out, dim=0, keepdim=True)[0]),dim=0)
        # res = self.fuse_conv(res.unsqueeze(0))[0]
        # res = self.a1*p_out + self.a2*c_out + self.a3*out
        # print(self.a1,self.a2,self.a3)
        # return p_out+c_outs
        return p_out

class AttenFusiona1(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusiona1, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)
        self.danet=DAModule(d_model=512,kernel_size=3)
    def forward(self, x):
        
    
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        a,b = self.danet(x)
        a,b = self.danet(a)
        x = b
        # print(x.shape)
        # exit()
        return x


# class AttenFusion(nn.Module):
#     def __init__(self, feature_dim):
#         super(AttenFusion, self).__init__()
#         self.att = ScaledDotProductAttention(feature_dim)

#     def forward(self, x):
        
    
#         cav_num, C, H, W = x.shape
#         x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
#         x = self.att(x, x, x)
#         x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
#         # print(x.shape)
#         # exit()
#         return x       

# class TransformerFusion(nn.Module):
#     def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
#         super(TransformerFusion, self).__init__()

#         self.att  = MultiHeadSelfAttention(channels,channels,channels,8)
#         self.cood = CoordAtt(5,5)

#     def forward(self, x):

#         cav_num, C, H, W = x.shape
#         if cav_num < 5:
#            padding =  torch.zeros([5-cav_num,C,H,W]).to(x.device)
#            x = torch.cat((x,padding),dim=0)
#         # print(x.shape)
#         # c, cav_num, H, W
#         x = x.permute(1,0,2,3)
#         # x = self.cood(x).permute(1,0,2,3)
#         x = self.cood(x).permute(1,0,2,3)[0]
#         # x = torch.max(x, dim=0, keepdim=False)[0]
#         # print(x.shape)
#         # exit()

#         return x

class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()

        self.communication = False
        self.round = 1
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = args['agg_operator']['mode']
        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']
            num_filters = args['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                if self.agg_mode == 'ATTEN':
                    # 暂时采用的是PositionAttentionModule，没有改动
                    fuse_network = AttenFusion(num_filters[idx])
                elif self.agg_mode == 'SF':    
                    fuse_network = SpatialFusion()
                elif self.agg_mode == 'Transformer':
                    fuse_network = TransformerFusion(
                                                channels=num_filters[idx], 
                                                n_head=args['agg_operator']['n_head'], 
                                                with_spe=args['agg_operator']['with_spe'], 
                                                with_scm=args['agg_operator']['with_scm'])
                self.fuse_modules.append(fuse_network)
        else:
            if self.agg_mode == 'ATTEN':
                self.fuse_modules = AttenFusion(args['agg_operator']['feature_dim'])
            elif self.agg_mode == 'SF':    
                 self.fuse_modules = SpatialFusion()
            elif self.agg_mode == 'Transformer':
                self.fuse_network = TransformerFusion(
                                            channels=args['agg_operator']['feature_dim'], 
                                            n_head=args['agg_operator']['n_head'], 
                                            with_spe=args['agg_operator']['with_spe'], 
                                            with_scm=args['agg_operator']['with_scm'])     

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

                                            
    def forward(self, x, record_len, backbone=None, heads=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B = len(record_len)
        
        # print("i======  0")
        if self.multi_scale:
            ups = []
            # backbone.__dict__()
            with_resnet = True if hasattr(backbone, 'resnet') else False
            # print(with_resnet)
            if with_resnet:
                feats = backbone.resnet(x)
            
            for i in range(self.num_levels):
                # 这里直接利用pointpillars里的backbone
                x = feats[i] if with_resnet else backbone.blocks[i](x)
                
                ############ 2. Split the confidence map #######################
                # split x:[(L1, C, H, W), (L2, C, H, W), ...]
                # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
                batch_node_features = self.regroup(x, record_len)
                # batch_node_features_bk = self.regroup(x_bk, record_len)
                ############ 3. Fusion ####################################
                x_fuse = []
                for b in range(B):
                    # number of valid agent
                    N = record_len[b]

                    neighbor_feature = batch_node_features[b]

                    x_fuse.append(self.fuse_modules[i](neighbor_feature))
                x_fuse = torch.stack(x_fuse)
                
                ############ 4. Deconv ####################################
                if len(backbone.deblocks) > 0:
                    ups.append(backbone.deblocks[i](x_fuse))
                else:
                    ups.append(x_fuse)

            if len(ups) > 1:
                # x_fuse = torch.cat(ups, dim=1)
                
                # 在这里增加了一个layer_fusion的融合方式，而不是简单的cat，具体代码在backbone里的LAM_Module_v2
                inp_fusion= torch.cat([ups[0].unsqueeze(1), ups[1].unsqueeze(1), ups[2].unsqueeze(1)], dim=1)
                # 1,3,128,100,352
                x_fuse = backbone.layer_fusion(inp_fusion)

            elif len(ups) == 1:
                x_fuse = ups[0]
            
            if len(backbone.deblocks) > self.num_levels:
                x_fuse = backbone.deblocks[-1](x_fuse)
        else:

            # with_resnet = True if hasattr(backbone, 'resnet') else False
            # # print(with_resnet)
            # if with_resnet:
            #     feats = backbone.resnet(x)

            ############ 1. Split the features #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                neighbor_feature = batch_node_features[b]

                x_fuse.append(self.fuse_modules(neighbor_feature))
            x_fuse = torch.stack(x_fuse)
        
        return x_fuse

