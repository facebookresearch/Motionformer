#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

"""Video models."""

from einops import rearrange, repeat
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch import einsum
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.hub import load_state_dict_from_url
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from . import performer_helper
from . import orthoformer_helper
from . import nystrom_helper

default_cfgs = {
    'vit_1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
    'vit_1k_large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
}


def qkv_attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class JointSpaceTimeAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, 
            self.num_heads, 
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Joint space-time attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class DividedAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # init to zeros
        self.qkv.weight.data.fill_(0)
        self.qkv.bias.data.fill_(0)
        self.proj.weight.data.fill_(1)
        self.proj.bias.data.fill_(0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        # num of heads variable
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Scale q
        q *= self.scale

        # Take out cls_q, cls_k, cls_v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), 
            (q_, k_, v_)
        )

        # expand CLS token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = qkv_attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_original_code=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # A typo in the original code meant that the value tensors for the temporal
        # attention step were identical to the input instead of being multiplied by a
        # learned projection matrix (v = x rather than v = Wx). The original code is
        # kept to facilitate replication, but is not recommended.
        self.use_original_code = use_original_code

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        B, N, C = x.shape
        P = seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # remove CLS token from q, k, v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v)
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)

        if approx == "nystrom":
            ## Shared spatial landmarks
            q_, k_, v_ = map(
                lambda t: rearrange(t, f'b h p d -> (b h) p d', h=h), (q_, k_, v_))
            x = nystrom_helper.nystrom_spatial_attn(
                q_, k_, v_,
                landmarks=num_landmarks,
                num_frames=F,
                inv_iters=6,
                use_spatial_landmarks=True
            )
            x = rearrange(x, f'(b h) p f d -> b h p f d', f=F, h=h)
        elif approx == "orthoformer":
            x = orthoformer_helper.orthoformer(
                q_, k_, v_,
                num_landmarks=num_landmarks,
                num_frames=F,
            )
        elif approx == "performer":
            # Form random projection matrices:
            m = 256 # r = 2m, m <= d
            d = self.head_dim
            seed = torch.ceil(torch.abs(torch.sum(q_) * performer_helper.BIG_CONSTANT))
            seed = torch.tensor(seed)
            projection_matrix = performer_helper.create_projection_matrix(
                m, d, seed=seed, device=q_.device, dtype=q_.dtype)
            q_, k_ = map(lambda t: rearrange(t, f'b h p d -> b p h d'), (q_, k_))
            q_prime = performer_helper.softmax_kernel_transformation(
                q_, 
                is_query=True, 
                projection_matrix=projection_matrix
            )
            k_prime = performer_helper.softmax_kernel_transformation(
                k_, 
                is_query=False, 
                projection_matrix=projection_matrix
            )
            q_prime, k_prime = map(
                lambda t: rearrange(t, f'b p h r -> b h p r'), (q_prime, k_prime))
            k_prime = rearrange(k_prime, 'b h (f n) r -> b h f n r', f=F)
            v_ = rearrange(v_, 'b h (f n) d -> b h f n d', f=F)
            kv = torch.einsum('b h f n r, b h f n d -> b h f r d', k_prime, v_)
            qkv = torch.einsum('b h p r, b h f r d -> b h p f d', q_prime, kv)
            normaliser = torch.einsum('b h f n r -> b h f r', k_prime)
            normaliser = torch.einsum('b h p r, b h f r -> b h p f', q_prime, normaliser)
            x = qkv / normaliser.unsqueeze(-1)
        else:
            # Using full attention
            q_dot_k = q_ @ k_.transpose(-2, -1)
            q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
            space_attn = (self.scale * q_dot_k).softmax(dim=-1)
            attn = self.attn_drop(space_attn)
            v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P)
            x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        if self.use_original_code:
            x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        else:
            x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def get_attention_module(
    attn_type='joint', dim=768, num_heads=12, qkv_bias=False, 
    attn_drop=0., proj_drop=0., use_original_code=True
):
    if attn_type == 'joint':
        attn = JointSpaceTimeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop)
    elif attn_type == 'trajectory':
        attn = TrajectoryAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_original_code=use_original_code)
    return attn


class Block(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory', 
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_original_code=True
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            attn_type=attn_type, dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_original_code=use_original_code
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x), 
                seq_len=seq_len, 
                num_frames=num_frames, 
                approx=approx,
                num_landmarks=num_landmarks
            )[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DividedSpaceTimeBlock(nn.Module):

    def __init__(
        self, dim=768, num_heads=12, attn_type='divided', 
        mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

        self.norm1 = norm_layer(dim)
        
        self.attn = DividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)

        self.timeattn = DividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

    def forward(self, x, seq_len=196, num_frames=8, approx='none', num_landmarks=128):
        time_output = self.timeattn(self.norm3(x), 
            self.einops_from_time, self.einops_to_time, n=seq_len)
        time_residual = x + time_output

        space_output = self.attn(self.norm1(time_residual), 
            self.einops_from_space, self.einops_to_space, f=num_frames)
        space_residual = time_residual + self.drop_path(space_output)

        x = space_residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, 
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size if type(img_size) is tuple else to_2tuple(img_size)
        patch_size = img_size if type(patch_size) is tuple else to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(
            self, img_size=224, temporal_resolution=4, in_chans=3, 
            patch_size=16, z_block_size=2, embed_dim=768, flatten=True
        ):
        super().__init__()
        self.height = (img_size // patch_size)
        self.width = (img_size // patch_size)
        self.frames = (temporal_resolution // z_block_size)
        self.num_patches = self.height * self.width * self.frames
        self.proj = nn.Conv3d(in_chans, embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size), 
            stride=(z_block_size, patch_size, patch_size))
        self.flatten = flatten

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class HeadMLP(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(HeadMLP, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=True),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )
        print(f"Dropout-NLP: {p}")

    def forward(self, x):
        return self.block_forward(x)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def adapt_input_conv(in_chans, conv_weight, agg='sum'):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            if agg == 'sum':
                print("Summing conv1 weights")
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
            else:
                print("Averaging conv1 weights")
                conv_weight = conv_weight.mean(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            if agg == 'sum':
                print("Summing conv1 weights")
                repeat = int(math.ceil(in_chans / 3))
                conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(in_chans))
            else:
                print("Averaging conv1 weights")
                conv_weight = conv_weight.mean(dim=1, keepdim=True)
                conv_weight = conv_weight.repeat(1, in_chans, 1, 1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(
    model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, progress=False
):
    # Load state dict
    assert(f"{cfg.VIT.PRETRAINED_WEIGHTS} not in [vit_1k, vit_1k_large]")
    state_dict = torch.hub.load_state_dict_from_url(url=default_cfgs[cfg.VIT.PRETRAINED_WEIGHTS])
    
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    input_convs = 'patch_embed.proj'
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(
                    in_chans, state_dict[weight_name], agg='avg')
                print(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                print(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifier_name = 'head'
    label_offset = cfg.get('label_offset', 0)
    pretrain_classes = 1000
    if num_classes != pretrain_classes:
        # completely discard fully connected if model num_classes doesn't match pretrained weights
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    elif label_offset > 0:
        # special case for pretrained weights with an extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    loaded_state = state_dict
    self_state = model.state_dict()
    all_names = set(self_state.keys())
    saved_names = set([])
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys() and param.shape == self_state[name].shape:
            saved_names.add(name)
            self_state[name].copy_(param)
        else:
            print(f"didnt load: {name} of shape: {param.shape}")
    print("Missing Keys:")
    print(all_names - saved_names)
