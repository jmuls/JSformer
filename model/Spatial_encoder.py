## Our model was revised from https://github.com/zczcwh/PoseFormer/blob/main/common/model_poseformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import DropPath

from common.opt import opts

opt = opts().parse()
device = torch.device("cuda")


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.edge_embedding = nn.Linear(17*17, 17*17)

    def forward(self, x, edge_embedding):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = edge_embedding.reshape(1, 17, 17).unsqueeze(0).repeat(B, self.num_heads, 1, 1)
        # print(edge_embedding.shape)

        attn = attn + edge_embedding


        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
class CVA_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.Qnorm = nn.LayerNorm(dim)
        self.Knorm = nn.LayerNorm(dim)
        self.Vnorm = nn.LayerNorm(dim)
        self.QLinear = nn.Linear(dim, dim)
        self.KLinear = nn.Linear(dim, dim)
        self.VLinear = nn.Linear(dim, dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.edge_embedding = nn.Linear(17*17, 17*17)




    def forward(self, x, CVA_input, edge_embedding):
        B, N, C = x.shape
        # CVA_input = self.max_pool(CVA_input)
        # print(CVA_input.shape)
        q = self.QLinear(self.Qnorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.KLinear(self.Knorm(CVA_input)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.VLinear(self.Vnorm(x)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        edge_embedding = self.edge_embedding(edge_embedding)
        edge_embedding = edge_embedding.reshape(1, 17, 17).unsqueeze(0).repeat(B, self.num_heads, 1, 1)



        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#######################################################################################################################
# 改进的注意力层（融合 hop 图和骨长图）
class AttentionWithHopBone(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可学习权重矩阵（标量参数）
        self.hop_weight = nn.Parameter(torch.ones(1))
        self.bone_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, hop_mat=None, bone_mat=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B, H, J, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, J, J]

        # 融入 hop 图和骨长图
        if hop_mat is not None:
            attn = attn + self.hop_weight * hop_mat.unsqueeze(1)  # [B,1,J,J] -> [B,H,J,J]
        if bone_mat is not None:
            attn = attn + self.bone_weight * bone_mat.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), edge_embedding))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#######################################################################################################################
class BlockWithHopBone(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # qkv (实现 attention 内部计算以便返回 MSA)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # drop path & mlp
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # hop-related modules to reproduce Multi_Out_Block behavior
        self.norm_hop1 = norm_layer(dim)
        self.norm_hop2 = norm_layer(dim)
        self.mlp_hop = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # --- 可选的跨视角交叉注意力（CVA）模块 ---
        self.cva_attn = CVA_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

    def forward(self, x, hops, hop_attn=None, bone_attn=None, CVA_input=None, edge_embedding=None):
        """
        x: (Bf, J, D)
        hops: (Bf, J, D) -- hop embedding for each joint
        hop_attn: (Bf, J, J) or None
        bone_attn: (Bf, J, J) or None
        CVA_input: (Bf, J, D) or None - 前一视角/外部提供的跨视角特征（可选）
        edge_embedding: (B, 17*17) or (17*17) or None - 传入 CVA 用的边嵌入（可选）
        returns: x, hops, MSA  (MSA has shape (Bf, J, D))
        """
        Bf, J, D = x.shape

        # 1) compute qkv & raw attn
        qkv = self.qkv(self.norm1(x)).reshape(Bf, J, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each: (Bf, H, J, d_head)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (Bf, H, J, J)

        # 2) add hop_attn / bone_attn if provided (expand to heads)
        if hop_attn is not None:
            # hop_attn shape expected (Bf, J, J)
            attn = attn + hop_attn.unsqueeze(1)  # -> (Bf, H, J, J)
        if bone_attn is not None:
            attn = attn + bone_attn.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3) attention output
        out = (attn @ v).transpose(1, 2).reshape(Bf, J, D)  # (Bf, J, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        # 4) match original Multi_Out_Block flow: MSA = drop_path(attn_out)
        MSA_self = self.drop_path(out)  # (Bf, J, D)

        # 5) modulate MSA by hop embedding: MSA = norm_hop1(hops) * MSA
        MSA_self = self.norm_hop1(hops) * MSA_self

        # 6) CVA 跨视角融合
        #    若同时提供 CVA_input 与 edge_embedding，则计算 MSA_cross 并融合
        if (CVA_input is not None) and (edge_embedding is not None):
            print("Using CVA in BlockWithHopBone")
            # 使用 MSA_self 作为 x 输入到 CVA（与你之前实现保持一致）
            # 注意 CVA_Attention 的 forward 签名为 (x, CVA_input, edge_embedding)
            MSA_cross = self.cva_attn(MSA_self, CVA_input, edge_embedding)  # (Bf, J, D)
            MSA = MSA_self + MSA_cross
        else:
            MSA = MSA_self

        # 7) update x and hops, then mlp's
        x = x + MSA
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        hops = hops + MSA
        hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))

        # return x, hops, MSA (MSA is same as used to update)
        return x, hops, MSA

# =================== BlockWithHopBone (返回 MSA) ===================
# class BlockWithHopBone(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)

#         # qkv (实现 attention 内部计算以便返回 MSA)
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(drop)

#         # drop path & mlp
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         # hop-related modules to reproduce Multi_Out_Block behavior
#         self.norm_hop1 = norm_layer(dim)
#         self.norm_hop2 = norm_layer(dim)
#         self.mlp_hop = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, hops, hop_attn=None, bone_attn=None):
#         """
#         x: (Bf, J, D)
#         hops: (Bf, J, D) -- hop embedding for each joint
#         hop_attn: (Bf, J, J) or None
#         bone_attn: (Bf, J, J) or None
#         returns: x, hops, MSA  (MSA has shape (Bf, J, D))
#         """
#         Bf, J, D = x.shape

#         # 1) compute qkv & raw attn
#         qkv = self.qkv(self.norm1(x)).reshape(Bf, J, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # each: (Bf, H, J, d_head)

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # (Bf, H, J, J)

#         # 2) add hop_attn / bone_attn if provided (expand to heads)
#         if hop_attn is not None:
#             attn = attn + hop_attn.unsqueeze(1)  # hop_attn: (Bf, J, J) -> (Bf,1,J,J) -> broadcast to heads
#         if bone_attn is not None:
#             attn = attn + bone_attn.unsqueeze(1)

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         # 3) attention output
#         out = (attn @ v).transpose(1, 2).reshape(Bf, J, D)  # (Bf, J, D)
#         out = self.proj(out)
#         out = self.proj_drop(out)

#         # 4) match original Multi_Out_Block flow: MSA = drop_path(attn_out)
#         MSA = self.drop_path(out)  # (Bf, J, D)

#         # 5) modulate MSA by hop embedding: MSA = norm_hop1(hops) * MSA
#         MSA = self.norm_hop1(hops) * MSA

#         # 6) update x and hops, then mlp's
#         x = x + MSA
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         hops = hops + MSA
#         hops = hops + self.drop_path(self.mlp_hop(self.norm_hop2(hops)))

#         # return x, hops, MSA (MSA is same as used to update)
#         return x, hops, MSA


# =================== First_view_Spatial_features_withHopBone ===================
class First_view_Spatial_features_withHopBone(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, hop_ch=68):
        """
        hop_ch: hop 特征的 channel 数（例如 68），如果不确定请设为 hop_feat.shape[1]
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_joints = num_joints
        self.num_frame = num_frame
        D = embed_dim_ratio

        # spatial embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, D)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, D))

        # hop feature embedding (per joint)
        self.hop_to_embedding = nn.Linear(hop_ch, D)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, D))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # use BlockWithHopBone stack (we keep 4 blocks as before by default)
        self.block1 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
                                       norm_layer=norm_layer)
        self.block2 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
                                       norm_layer=norm_layer)
        self.block3 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2],
                                       norm_layer=norm_layer)
        self.block4 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3],
                                       norm_layer=norm_layer)

        self.Spatial_norm = norm_layer(D)
        self.hop_norm = norm_layer(D)

    def _prepare_bone_mat(self, bone_mat, B, F):
        if bone_mat is None:
            return None
        if bone_mat.dim() == 3 and bone_mat.shape[0] == B:
            return rearrange(bone_mat.unsqueeze(1).repeat(1, F, 1, 1), 'b f j1 j2 -> (b f) j1 j2')
        if bone_mat.dim() == 4 and bone_mat.shape[0] == B and bone_mat.shape[1] == F:
            return rearrange(bone_mat, 'b f j1 j2 -> (b f) j1 j2')
        if bone_mat.dim() == 3 and bone_mat.shape[0] == B * F:
            return bone_mat
        raise ValueError(f"Unsupported bone_mat shape {tuple(bone_mat.shape)} for B={B},F={F}")

    def forward(self, x, hop_feat, bone_mat=None):
        """
        x: [B, C, F, J]
        hop_feat: [B, C_hop, F, J]
        bone_mat: optional (B,J,J) or (B,F,J,J) or (B*F,J,J)
        returns: x_out (B,F,J*D), hop_out (B,F,J*D), MSA1..MSA4 each (B,F,J,D) flattened as needed by caller
        """
        B, C, F, J = x.shape
        assert J == self.num_joints and F == self.num_frame

        # --- process x into per-frame joint embeddings ---
        x_flat = rearrange(x, 'b c f p -> (b f) p c')          # (Bf, J, C)
        x_flat = self.Spatial_patch_to_embedding(x_flat)       # (Bf, J, D)
        x_flat = x_flat + self.Spatial_pos_embed
        x_flat = self.pos_drop(x_flat)

        # --- process hop_feat to embeddings ---
        hop_flat = rearrange(hop_feat, 'b ch f p -> (b f) p ch')  # (Bf, J, C_hop)
        hop_emb = self.hop_to_embedding(hop_flat)                 # (Bf, J, D)
        hop_emb = hop_emb + self.hop_pos_embed
        hop_emb = self.pos_drop(hop_emb)

        # --- build hop attention matrices (Bf, J, J) from hop_emb ---
        hop_attn = torch.matmul(hop_emb, hop_emb.transpose(-2, -1))  # (Bf, J, J)
        hop_attn = hop_attn / (torch.clamp(hop_attn.abs().sum(-1, keepdim=True), min=1e-6))

        bone_attn = self._prepare_bone_mat(bone_mat, B, F) if bone_mat is not None else None

        # --- pass through blocks, collecting MSA outputs ---
        x1, hop1, MSA1 = self.block1(x_flat, hop_emb, hop_attn, bone_attn)
        x2, hop2, MSA2 = self.block2(x1, hop1, hop_attn, bone_attn)
        x3, hop3, MSA3 = self.block3(x2, hop2, hop_attn, bone_attn)
        x4, hop4, MSA4 = self.block4(x3, hop3, hop_attn, bone_attn)

        # normalize & reshape back to (B, F, J*D)
        x_out = self.Spatial_norm(x4)                           # (Bf, J, D)
        x_out = rearrange(x_out, '(b f) j d -> b f (j d)', b=B, f=F)

        hop_out = self.hop_norm(hop4)                           # (Bf, J, D)
        hop_out = rearrange(hop_out, '(b f) j d -> b f (j d)', b=B, f=F)

        # reshape MSAs to (B, F, J, D) to match earlier interface expectations
        MSA1 = rearrange(MSA1, '(b f) j d -> b f j d', b=B, f=F)
        MSA2 = rearrange(MSA2, '(b f) j d -> b f j d', b=B, f=F)
        MSA3 = rearrange(MSA3, '(b f) j d -> b f j d', b=B, f=F)
        MSA4 = rearrange(MSA4, '(b f) j d -> b f j d', b=B, f=F)

        return x_out, hop_out, MSA1, MSA2, MSA3, MSA4


# =================== Spatial_features_withHopBone (后续视角) ===================
class Spatial_features_withHopBone(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None, hop_ch=68,use_cva=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_joints = num_joints
        self.num_frame = num_frame
        D = embed_dim_ratio

        self.use_cva = use_cva

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, D)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, D))

        self.hop_to_embedding = nn.Linear(hop_ch, D)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, D))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.block1 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                       attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.block2 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                       attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer)
        self.block3 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                       attn_drop=attn_drop_rate, drop_path=dpr[2], norm_layer=norm_layer)
        self.block4 = BlockWithHopBone(dim=D, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                       attn_drop=attn_drop_rate, drop_path=dpr[3], norm_layer=norm_layer)


        self.Spatial_norm = norm_layer(D)
        self.hop_norm = norm_layer(D)

    def _prepare_bone_mat(self, bone_mat, B, F):
        if bone_mat is None:
            return None
        if bone_mat.dim() == 3 and bone_mat.shape[0] == B:
            return rearrange(bone_mat.unsqueeze(1).repeat(1, F, 1, 1), 'b f j1 j2 -> (b f) j1 j2')
        if bone_mat.dim() == 4 and bone_mat.shape[0] == B and bone_mat.shape[1] == F:
            return rearrange(bone_mat, 'b f j1 j2 -> (b f) j1 j2')
        if bone_mat.dim() == 3 and bone_mat.shape[0] == B * F:
            return bone_mat
        raise ValueError(f"Unsupported bone_mat shape {tuple(bone_mat.shape)} for B={B},F={F}")

    def forward(self, x, hop_feat, MSA1, MSA2, MSA3, MSA4, bone_mat=None):
        """
        x: [B, C, F, J]
        hop_feat: [B, C_hop, F, J]
        MSA1..MSA4: previous-stage attention outputs (B, F, J, D) -- here we ignore direct usage but kept interface
        bone_mat: optional
        returns: x_out, hop_out, MSA1..MSA4 (updated)
        """
        B, C, F, J = x.shape
        assert J == self.num_joints and F == self.num_frame

        x_flat = rearrange(x, 'b c f p -> (b f) p c')   # (Bf, J, C)
        x_flat = self.Spatial_patch_to_embedding(x_flat)
        x_flat = x_flat + self.Spatial_pos_embed
        x_flat = self.pos_drop(x_flat)

        hop_flat = rearrange(hop_feat, 'b ch f p -> (b f) p ch')
        hop_emb = self.hop_to_embedding(hop_flat)
        hop_emb = hop_emb + self.hop_pos_embed
        hop_emb = self.pos_drop(hop_emb)

        # hop attention matrix
        hop_attn = torch.matmul(hop_emb, hop_emb.transpose(-2, -1))
        hop_attn = hop_attn / (torch.clamp(hop_attn.abs().sum(-1, keepdim=True), min=1e-6))

        bone_attn = self._prepare_bone_mat(bone_mat, B, F) if bone_mat is not None else None

        # pass through blocks; collect MSAs
        x1, hop1, M1 = self.block1(x_flat, hop_emb, hop_attn, bone_attn)
        x2, hop2, M2 = self.block2(x1, hop1, hop_attn, bone_attn)
        x3, hop3, M3 = self.block3(x2, hop2, hop_attn, bone_attn)
        x4, hop4, M4 = self.block4(x3, hop3, hop_attn, bone_attn)


        x_out = self.Spatial_norm(x4)
        x_out = rearrange(x_out, '(b f) j d -> b f (j d)', b=B, f=F)

        hop_out = self.hop_norm(hop4)
        hop_out = rearrange(hop_out, '(b f) j d -> b f (j d)', b=B, f=F)

        # reshape MSAs to (B,F,J,D)
        M1 = rearrange(M1, '(b f) j d -> b f j d', b=B, f=F)
        M2 = rearrange(M2, '(b f) j d -> b f j d', b=B, f=F)
        M3 = rearrange(M3, '(b f) j d -> b f j d', b=B, f=F)
        M4 = rearrange(M4, '(b f) j d -> b f j d', b=B, f=F)

        return x_out, hop_out, M1, M2, M3, M4
