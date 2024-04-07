default_scope = 'mmpose'

backend_interval = 600
save_interval = 10
no_samples = 5

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=backend_interval),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='VisdomCheckpointHook', interval=save_interval),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='RandomPoseVisualizationHook', no_samples=no_samples),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        metric_type='loss',
        badcase_thr=5,
        interval=backend_interval))

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='VisdomBackend', init_kwargs=dict(name="vitpose", server="localhost", port=8097, env="test_vitpose"))
]
visualizer = dict(
    type='VisdomPoseVisualizer', vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(type='VisdomLogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')

# training/validation/testing progress
train_cfg = dict(by_epoch=True)
val_cfg = dict()