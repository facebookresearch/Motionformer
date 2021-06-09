#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import math


def orthogonal_landmarks(q, k, num_landmarks=64, subsample_fraction=1.0):
    """
    Construct set of landmarks by recursively selecting new landmarks 
    that are maximally orthogonal to the existing set.
    Returns near orthogonal landmarks with shape (B, M, D).
    """
    if subsample_fraction < 1.0:
        # Need at least M/2 samples of queries and keys
        num_samples = max(int(subsample_fraction * q.size(-2)), num_landmarks)
        q_unnormalised = q[:, torch.randint(q.size(-2), (num_samples,), device=q.device), :] # (B, N, D)
    else:
        # (B, N, D)
        q_unnormalised = q

    # may need to change default eps to eps=1e-8 for mixed precision compatibility
    qk = Fn.normalize(q_unnormalised, p=2, dim=-1)
    B, N, D = qk.shape

    selected_mask = torch.zeros((B, N, 1), device=qk.device)
    landmark_mask = torch.ones((B, 1, 1), dtype=selected_mask.dtype, device=qk.device)

    # Get initial random landmark
    random_idx = torch.randint(qk.size(-2), (B, 1, 1), device=qk.device)
    selected_landmark = qk[torch.arange(qk.size(0)), random_idx.view(-1), :].view(B, D)
    selected_mask.scatter_(-2, random_idx, landmark_mask)

    # Selected landmarks
    selected_landmarks = torch.empty((B, num_landmarks, D), device=qk.device, dtype=qk.dtype)
    selected_landmarks[:, 0, :] = selected_landmark

    # Store computed cosine similarities
    cos_sims = torch.empty((B, N, num_landmarks), device=qk.device, dtype=qk.dtype)

    for M in range(1, num_landmarks):
        # Calculate absolute cosine similarity between selected and unselected landmarks
        # (B, N, D) * (B, D) -> (B, N)
        cos_sim = torch.einsum('b n d, b d -> b n', qk, selected_landmark).abs()
        cos_sims[:, :, M - 1] = cos_sim
        # (B, N, M) cosine similarities of current set of landmarks wrt all queries and keys
        cos_sim_set = cos_sims[:, :, :M]

        # Get orthogonal landmark: landmark with smallest absolute cosine similarity:
        # set cosine similarity for already selected landmarks to > 1
        cos_sim_set.view(-1, M)[selected_mask.flatten().bool(), :] = 10
        # (B,) - want max for non
        selected_landmark_idx = cos_sim_set.amax(-1).argmin(-1)
        selected_landmark = qk[torch.arange(qk.size(0)), selected_landmark_idx, :].view(B, D)

        # Add most orthogonal landmark to selected landmarks: 
        selected_landmarks[:, M, :] = selected_landmark

        # Removed selected indices from non-selected mask: 
        selected_mask.scatter_(-2, selected_landmark_idx.unsqueeze(-1).unsqueeze(-1), landmark_mask)
    landmarks = torch.masked_select(
        q_unnormalised, selected_mask.bool()).reshape(B, -1, D) # (B, M, D)
    return landmarks # (B, M, D)


def orthoformer(
    q, k, v, num_landmarks=64, subsample_fraction=1.0, 
    num_frames=None, shared_landmarks=True, return_attn=False
):
    """
    Computes spatial attention for all pairs of frames.
    The attention matrix is approximated using 
    intermediate landmarks taken from the queries and keys.
    The landmarks can be unique (to each frame) or 
    shared (a common set of landmarks across frames).
    """
    B, N, D = k.shape
    F = num_frames
    L = num_landmarks
    P = N // F

    scale = D ** -0.25
    q = q * scale
    k = k * scale
    
    if shared_landmarks:
        with torch.no_grad():
            landmarks = orthogonal_landmarks(q, k, num_landmarks, subsample_fraction)
        kernel_1 = Fn.softmax(torch.matmul(q, landmarks.transpose(-1, -2)), dim=-1)
        kernel_2 = Fn.softmax(
            rearrange(torch.matmul(
                landmarks, k.transpose(-1, -2)), 'b l (f p) -> b l f p', f=F), dim=-1)
        v = rearrange(v, 'b (f p) d -> b f p d', f=F)
        x = torch.einsum('b l f p, b f p d -> b l f d', kernel_2, v)
        x = torch.einsum('b n l, b l f d -> b n f d', kernel_1, x)
        if return_attn:
            attn = torch.einsum('b m l, b l f p -> b m f p', kernel_1, kernel_2)
            return x, attn
    else:
        q = rearrange(q, 'b (f p) d -> (b f) p d', f=F)
        k = rearrange(k, 'b (g q) d -> (b g) q d', g=F)
        with torch.no_grad():
            landmarks = orthogonal_landmarks(q, k, num_landmarks, subsample_fraction)
            landmarks = rearrange(landmarks, '(b f) l d -> b f l d', f=F)
        q = rearrange(q, '(b f) p d -> b f 1 p d', f=F)
        k = rearrange(k, '(b g) q d -> b 1 g q d', g=F)
        v = rearrange(v, 'b (g q) d -> b 1 g q d', g=F)
        kernel_1 = Fn.softmax(
            torch.matmul(q, landmarks.unsqueeze(-4).transpose(-1, -2)), dim=-1)
        kernel_2 = Fn.softmax(
            torch.matmul(landmarks.unsqueeze(-3), k.transpose(-1, -2)), dim=-1)
        x = torch.matmul(kernel_1, torch.matmul(kernel_2, v))
        x = rearrange(x, 'b f g p d -> b (f p) g d')
        if return_attn:
            attn = torch.matmul(kernel_1, kernel_2)
            attn = rearrange(attn, 'b f g p q -> b (f p) g q')
            return x, attn

    return x