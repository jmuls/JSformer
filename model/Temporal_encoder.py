## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange
from timm.layers import DropPath
from scipy.fftpack import dct, idct
from common.opt import opts

opt = opts().parse()
device = torch.device("cuda")


######################### 同步对称矩阵定义 #########################
def get_symmetry_graph(num_joints=17, device='cuda'):
    sym_graph = torch.eye(num_joints, device=device)
    # 对称肩肘
    sym_graph[3,6] = sym_graph[6,3] = 1
    sym_graph[4,7] = sym_graph[7,4] = 1
    
    # 对称髋膝踝
    sym_graph[9,12] = sym_graph[12,9] = 1
    sym_graph[10,13] = sym_graph[13,10] = 1
    sym_graph[11,14] = sym_graph[14,11] = 1

    # 左手肘与右脚膝同步
    sym_graph[4,13] = sym_graph[13,4] = 1
    sym_graph[7,10] = sym_graph[10,7] = 1

    # 右手腕与左脚踝同步
    sym_graph[8,5] = sym_graph[5,8] = 1
    sym_graph[11,14] = sym_graph[14,11] = 1

    return sym_graph



#######################################################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


#######################################################################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., num_joints=17,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#######################################################################################################################
# 同步对称图特征增强层（时序编码器输入前使用）
class SymmetrySyncGraph(nn.Module):
    def __init__(self, num_joints, embed_dim):
        super().__init__()
        self.num_joints = num_joints
        self.emb_dim = embed_dim
        self.sym_weight = nn.Parameter(torch.randn(num_joints, num_joints))
        self.linear = nn.Linear(embed_dim, embed_dim)  # 类似 GCN 的 W
        self.act = nn.GELU()
        

    def forward(self, x, sym_graph):
        if sym_graph is None:
            sym_graph = get_symmetry_graph(self.num_joints, device=x.device)
        B, T, JC = x.shape
        C = JC // self.num_joints
        x = x.view(B, T, self.num_joints, C)

        sym_graph = sym_graph * self.sym_weight
        sym_graph = torch.softmax(sym_graph, dim=-1)
        x_new = torch.einsum('ij,btjc->btic', sym_graph, x)
        x_new = self.linear(x_new)
        x = x + self.act(x_new)
        return x.view(B, T, -1)

#######################################################################################################################
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,sym_graph=None):
        
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#######################################################################################################################
class Temporal__features(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        
        self.num_joints = num_joints
        self.sym_enhance = SymmetrySyncGraph(num_joints, embed_dim_ratio)
        

        ### Temporal patch embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Temporal_norm = norm_layer(embed_dim)
        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

    def forward(self, x):
        b = x.shape[0]
        # x_residual = x
        x = self.sym_enhance(x, sym_graph=None)
        # x = x + x_residual
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        # x = self.weighted_mean(x)
        x = x.view(b, opt.frames, -1)
        return x