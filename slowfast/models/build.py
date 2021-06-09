#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model construction functions."""
import math
import torch
import slowfast as slowfast
from fvcore.common.registry import Registry

from . import vit_helper

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)

    if isinstance(model, slowfast.models.video_model_builder.VisionTransformer):
        if cfg.VIT.IM_PRETRAINED:
            vit_helper.load_pretrained(
                model, cfg=cfg, num_classes=cfg.MODEL.NUM_CLASSES, 
                in_chans=cfg.VIT.CHANNELS, filter_fn=vit_helper._conv_filter, 
                strict=False
            )
            if hasattr(model, 'st_embed'):
                model.st_embed.data[:, 1:, :] = model.pos_embed.data[:, 1:, :].repeat(
                    1, cfg.VIT.TEMPORAL_RESOLUTION, 1)
                model.st_embed.data[:, 0, :] = model.pos_embed.data[:, 0, :]
            if hasattr(model, 'patch_embed_3d'):
                model.patch_embed_3d.proj.weight.data = torch.zeros_like(
                    model.patch_embed_3d.proj.weight.data)
                n = math.floor(model.patch_embed_3d.proj.weight.shape[2] / 2)
                model.patch_embed_3d.proj.weight.data[:, :, n, :, :] = model.patch_embed.proj.weight.data
                model.patch_embed_3d.proj.bias.data = model.patch_embed.proj.bias.data

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=True
        )
    return model
