# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    'mmaction::_base_/models/tsn_r50.py', 'mmaction::_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'

file_client_args = dict(io_backend='disk')

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=4),  # Adjust clip_len and num_clips
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=6,
    num_workers=48,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = []

test_cfg = dict(type='TestLoop')