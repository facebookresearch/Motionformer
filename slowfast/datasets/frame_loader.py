#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import torch
from . import utils as utils
from .decoder import get_start_end_idx


def temporal_sampling(
    num_frames, start_idx, end_idx, num_samples, start_frame=0
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def pack_frames_to_video_clip(
    cfg, video_record, temporal_sample_index, target_fps=60
):
    # Load video by loading its extracted frames
    path_to_video = '{}/{}/rgb_frames/{}'.format(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR,
        video_record.participant,
        video_record.untrimmed_video_name
    
    )
    img_tmpl = "frame_{:010d}.jpg"
    fps = video_record.fps
    sampling_rate = cfg.DATA.SAMPLING_RATE
    num_samples = cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(
        video_record.num_frames,
        start_idx, end_idx, num_samples,
        start_frame=video_record.start_frame
    )
    img_paths = [
        os.path.join(
            path_to_video, 
            img_tmpl.format(idx.item()
        )) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths)
    return frames