default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html

default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
    runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
    timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
    logger=dict(
        type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
        interval=20,  # Interval to print the log
        ignore_last=False), # Ignore the log of last iterations in each epoch
    param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
    checkpoint=dict(
        type='CheckpointHook',  # The hook to save checkpoints periodically
        interval=1,  # The saving period
        save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
        max_keep_ckpts=3),  # The maximum checkpoints to keep
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
    sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch

env_cfg = dict(  # Dict for setting environment
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # Parameters to setup multiprocessing
    dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

log_processor = dict(
    type='LogProcessor',  # Log processor used to format log information
    window_size=20,  # Default smooth interval
    by_epoch=True)  # Whether to format logs with epoch type

vis_backends = [  # List of visualization backends
    dict(type='LocalVisBackend')]  # Local visualization backend
visualizer = dict(  # Config of visualizer
    type='ActionVisualizer',  # Name of visualizer
    vis_backends=vis_backends)

log_level = 'INFO'  # The level of logging
load_from = None  # The path to the checkpoint to load from. It can be a file path, URL, or None. If set to None, the model will not load any checkpoints at the beginning. If set to a URL, the checkpoint will be downloaded and cached before loading. If set to a file path, the checkpoint will be directly loaded.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.