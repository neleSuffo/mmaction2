_base_ = [
    '../../_base_/models/bmn_400x100.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = '/home/nele_pauline_suffo/ProcessedData/bmn_childlens/mmaction_feat'
ann_file_train = '/home/nele_pauline_suffo/ProcessedData/bmn_childlens/anno_train.json'
ann_file_val = '/home/nele_pauline_suffo/ProcessedData/bmn_childlens/anno_val.json'
ann_file_test = '/home/nele_pauline_suffo/ProcessedData/bmn_childlens/anno_test.json'

train_pipeline = [ # Training data processing pipeline
    dict(type='LoadLocalizationFeature'), # Load localization feature pipeline
    dict(type='GenerateLocalizationLabels'), # Generate localization labels pipeline
    dict(
        type='PackLocalizationInputs', # Pack localization data
        keys=('gt_bbox', ), # Keys of input
        meta_keys=('video_name', )) # Meta keys of input
]

val_pipeline = [ # Validation data processing pipeline
    dict(type='LoadLocalizationFeature'), # Load localization feature pipeline
    dict(type='GenerateLocalizationLabels'), # Generate localization labels pipeline
    dict(
        type='PackLocalizationInputs', # Pack localization data
        keys=('gt_bbox', ), # Keys of input
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame')) # Meta keys of input
]

test_pipeline = [ # Testing data processing pipeline
    dict(type='LoadLocalizationFeature'), # Load localization feature pipeline
    dict(
        type='PackLocalizationInputs', # Pack localization data
        keys=('gt_bbox', ), # Keys of input
        meta_keys=('video_name', 'duration_second', 'duration_frame',
                   'annotations', 'feature_frame')) # Meta keys of input
]

train_dataloader = dict( # Config of train data loader
    batch_size=4, # Batch size of each single GPU during training
    num_workers=48, # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True, # if "True", the data loader will not shutdown the worker processes after an epoch end, which can accelerate training speed 
    sampler=dict(
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
        shuffle=True),  # Randomly shuffle the training data in each epoch
    drop_last=True,
    dataset=dict(  # Config of train dataset
        type=dataset_type,
        ann_file=ann_file_train,  # Path of annotation file
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(  # Config of validation dataloader
    batch_size=1,  # Batch size of each single GPU during evaluation
    num_workers=48,  # Workers to pre-fetch data for each single GPU during evaluation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler', 
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of validation dataset
        type=dataset_type,
        ann_file=ann_file_val,  # Path of annotation file
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(  # Config of test dataloader
    batch_size=1,  # Batch size of each single GPU during testing
    num_workers=48,  # Workers to pre-fetch data for each single GPU during testing
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(
        type='DefaultSampler', 
        shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of test dataset
        type=dataset_type,
        ann_file=ann_file_test, # Path of annotation file
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True))

max_epochs = 100  # Total epochs to train the model
train_cfg = dict(  # Config of training loop
    type='EpochBasedTrainLoop',  # Name of training loop
    max_epochs=max_epochs,  # Total training epochs
    val_begin=1,  # The epoch that begins validating
    val_interval=1)  # Validation interval

val_cfg = dict(  # Config of validation loop
    type='ValLoop')  # Name of validating loop
test_cfg = dict( # Config of testing loop
    type='TestLoop')  # Name of testing loop

# optimizer
optim_wrapper = dict(  # Config of optimizer wrapper
    type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='Adam',  # Name of optimizer
        lr=0.001,  # Learning rate
        weight_decay=0.0001),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip

# learning policy
param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
    dict(type='MultiStepLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
    begin=0,  # Step at which to start updating the learning rate
    end=max_epochs,  # Step at which to stop updating the learning rate
    by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
    milestones=[7, ],  # Steps to decay the learning rate
    gamma=0.1)]  # Multiplicative factor of parameter value decay

work_dir = './work_dirs/bmn_400x100_2x8_9e_childlens_feature/'
test_evaluator = dict(
    type='ANetMetric',
    metric_type='AR@AN',
    dump_config=dict(out=f'{work_dir}/results.json', output_format='json'))
val_evaluator = test_evaluator

# Set PYTORCH_CUDA_ALLOC_CONF to manage memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
