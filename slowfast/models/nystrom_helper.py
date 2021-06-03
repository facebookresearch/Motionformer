#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import math


def iterative_inv(mat, n_iter = 6, init_option="exact"):
    I = torch.eye(mat.size(-2), device = mat.device)
    K = mat

    if init_option == "original":
        # This original implementation is more conservative to compute coefficient of Z_0. 
        V = 1. / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
    elif init_option == "arbitrary_input":
        # sum = 1 for softmax input but not for exp
        a1 = torch.max(torch.sum(torch.abs(K), dim = -2, keepdim=True), dim=-1, keepdim=True).values
        a2 = torch.max(torch.sum(torch.abs(K), dim = -1, keepdim=True), dim=-2, keepdim=True).values
        V = 1. / (a1 * a2) * K.transpose(-1, -2)
    else: # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        # This is the exact coefficient computation, 
        # 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
        V = 1. / torch.max(
            torch.sum(K, dim = -2), dim = -1).values.unsqueeze(-1).unsqueeze(-1) * K.transpose(-1, -2)

    for _ in range(n_iter):
        KV = torch.matmul(K, V)
        V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
    return V


def nystrom_spatial_attn(
    q, k, v, landmarks=64, num_frames=None, inv_iters=6, 
    use_full_matrix=False, use_spatial_landmarks=False, return_attn=False
):

    """
    Compute full space-time attention but only softmax over spatial dimension
    """
    B, N, D = k.shape
    F = num_frames
    scale = D ** -0.5
    q = q * scale
    if use_full_matrix:
        queries_landmarks = q.clone()
        keys_landmarks = k.clone()
    else:
        segs = N // landmarks
        with torch.no_grad():
            if use_spatial_landmarks:
                # transpose spatial and temporal dimensions
                q2 = rearrange(q, 'b (f p) d -> b (p f) d', f=F)
                k2 = rearrange(k, 'b (f p) d -> b (p f) d', f=F)
                if (N % landmarks == 0):
                    keys_landmarks = k2.reshape(B, landmarks, N // landmarks, D).mean(dim = -2)
                    queries_landmarks = q2.reshape(B, landmarks, N // landmarks, D).mean(dim = -2)
                else:
                    num_k = (segs + 1) * landmarks - N
                    keys_landmarks_f = k2[:, :num_k * segs, :].reshape(
                        B, num_k, segs, D).mean(dim = -2)
                    keys_landmarks_l = k2[:, num_k * segs:, :].reshape(
                        B, landmarks - num_k, segs + 1, D).mean(dim = -2)
                    keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

                    queries_landmarks_f = q2[:, :num_k * segs, :].reshape(
                        B, num_k, segs, D).mean(dim = -2)
                    queries_landmarks_l = q2[:, num_k * segs:, :].reshape(
                        B, landmarks - num_k, segs + 1, D).mean(dim = -2)
                    queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)
            else:
                if (N % landmarks == 0):
                    keys_landmarks = k.reshape(
                        B, landmarks, N // landmarks, D).mean(dim = -2)
                    queries_landmarks = q.reshape(
                        B, landmarks, N // landmarks, D).mean(dim = -2)
                else:
                    num_k = (segs + 1) * landmarks - N
                    keys_landmarks_f = k[:, :num_k * segs, :].reshape(
                        B, num_k, segs, D).mean(dim = -2)
                    keys_landmarks_l = k[:, num_k * segs:, :].reshape(
                        B, landmarks - num_k, segs + 1, D).mean(dim = -2)
                    keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

                    queries_landmarks_f = q[:, :num_k * segs, :].reshape(
                        B, num_k, segs, D).mean(dim = -2)
                    queries_landmarks_l = q[:, num_k * segs:, :].reshape(
                        B, landmarks - num_k, segs + 1, D).mean(dim = -2)
                    queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)

    kernel_1 = Fn.softmax(
        torch.matmul(q, keys_landmarks.transpose(-1, -2)), dim = -1)
    kernel_2 = Fn.softmax(
        torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
    kernel_3 = Fn.softmax(
        rearrange(torch.matmul(
            queries_landmarks, k.transpose(-1, -2)), 'b l (f p) -> b l f p', f=F), dim = -1)
    attn = torch.matmul(kernel_1, iterative_inv(kernel_2, n_iter=inv_iters))

    v = rearrange(v, 'b (f p) d -> b f p d', f=F)
    x = torch.einsum(
        'b n l, b l f d -> b n f d', 
        attn, torch.einsum('b l f p, b f p d -> b l f d', kernel_3, v)
    )

    if return_attn:
        attn = torch.einsum('b m l, b l f p -> b m f p', attn, kernel_3)
        return x, attn

    return x