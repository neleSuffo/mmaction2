# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import config
import torch
from mmengine import dump, list_from_file, load
from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 feature extraction')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_prefix', type=str, help='output prefix')
    parser.add_argument(
        '--video-list', type=str, default=None, help='video file list')
    parser.add_argument(
        '--video-root', type=str, default=None, help='video root directory')
    parser.add_argument(
        '--spatial-type',
        type=str,
        default='avg',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in spatial dimension')
    parser.add_argument(
        '--temporal-type',
        type=str,
        default='avg',
        choices=['avg', 'max', 'keep'],
        help='Pooling type in temporal dimension')
    parser.add_argument(
        '--long-video-mode',
        action='store_true',
        help='Perform long video inference to get a feature list from a video')
    parser.add_argument(
        '--clip-interval',
        type=int,
        default=None,
        help='Clip interval for Clip interval of adjacent center of sampled '
        'clips, used for long video inference')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=None,
        help='Temporal interval of adjacent sampled frames, used for long '
        'video long video inference')
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Perform multi view inference')
    parser.add_argument(
        '--dump-score',
        action='store_true',
        help='Dump predict scores rather than features')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    # -------------------- Feature Head --------------------
    if not args.dump_score:
        backbone_type2name = dict(
            ResNet3dSlowFast='slowfast',
            MobileNetV2TSM='tsm',
            ResNetTSM='tsm',
        )

        if cfg.model.type == 'RecognizerGCN':
            backbone_name = 'gcn'
        else:
            backbone_name = backbone_type2name.get(cfg.model.backbone.type)
        num_segments = None
        if backbone_name == 'tsm':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'UntrimmedSampleFrames':
                    clip_len = transform['clip_len']
                    continue
                elif transform.type == 'SampleFrames':
                    clip_len = transform['num_clips']
            num_segments = cfg.model.backbone.get('num_segments', 8)
            assert num_segments == clip_len, \
                f'num_segments and clip length must same for TSM, but got ' \
                f'num_segments {num_segments} clip_len {clip_len}'
            if cfg.model.test_cfg is not None:
                max_testing_views = cfg.model.test_cfg.get(
                    'max_testing_views', num_segments)
                assert max_testing_views % num_segments == 0, \
                    'tsm needs to infer with batchsize of multiple ' \
                    'of num_segments.'

        spatial_type = None if args.spatial_type == 'keep' else \
            args.spatial_type
        temporal_type = None if args.temporal_type == 'keep' else \
            args.temporal_type
        feature_head = dict(
            type='FeatureHead',
            spatial_type=spatial_type,
            temporal_type=temporal_type,
            backbone_name=backbone_name,
            num_segments=num_segments)
        cfg.model.cls_head = feature_head

    # ---------------------- multiple view ----------------------
    if not args.multi_view:
        # average features among multiple views
        cfg.model.cls_head['average_clips'] = 'score'
        if cfg.model.type == 'Recognizer3D':
            for idx, transform in enumerate(test_pipeline):
                if transform.type == 'SampleFrames':
                    test_pipeline[idx]['num_clips'] = 1
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'SampleFrames':
                test_pipeline[idx]['twice_sample'] = False
            # if transform.type in ['ThreeCrop', 'TenCrop']:
            if transform.type == 'TenCrop':
                test_pipeline[idx].type = 'CenterCrop'

    # -------------------- pipeline settings  --------------------
    # assign video list and video root
    if args.video_list is not None:
        cfg.test_dataloader.dataset.ann_file = args.video_list
    if args.video_root is not None:
        if cfg.test_dataloader.dataset.type == 'VideoDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(
                video=args.video_root)
        elif cfg.test_dataloader.dataset.type == 'RawframeDataset':
            cfg.test_dataloader.dataset.data_prefix = dict(img=args.video_root)
    args.video_list = cfg.test_dataloader.dataset.ann_file
    args.video_root = cfg.test_dataloader.dataset.data_prefix
    # use UntrimmedSampleFrames for long video inference
    if args.long_video_mode:
        config.logging.info('Performing long video inference.')
        # preserve features of multiple clips
        cfg.model.cls_head['average_clips'] = None
        cfg.test_dataloader.batch_size = 1
        is_recognizer2d = (cfg.model.type == 'Recognizer2D')

        frame_interval = args.frame_interval
        for idx, transform in enumerate(test_pipeline):
            if transform.type == 'UntrimmedSampleFrames':
                clip_len = transform['clip_len']
                continue
            # replace SampleFrame by UntrimmedSampleFrames
            elif transform.type in ['SampleFrames', 'UniformSample']:
                assert args.clip_interval is not None, \
                    'please specify clip interval for long video inference'
                if is_recognizer2d:
                    # clip_len of UntrimmedSampleFrames is same as
                    # num_clips for 2D Recognizer.
                    clip_len = transform['num_clips']
                else:
                    clip_len = transform['clip_len']
                    if frame_interval is None:
                        # take frame_interval of SampleFrames as default
                        frame_interval = transform.get('frame_interval')
                assert frame_interval is not None, \
                    'please specify frame interval for long video ' \
                    'inference when use UniformSample or 2D Recognizer'

                sample_cfgs = dict(
                    type='UntrimmedSampleFrames',
                    clip_len=clip_len,
                    clip_interval=args.clip_interval,
                    frame_interval=frame_interval)
                test_pipeline[idx] = sample_cfgs
                continue
        # flow input will stack all frames
        if cfg.test_dataloader.dataset.get('modality') == 'Flow':
            clip_len = 1

        if is_recognizer2d:
            from mmaction.models import ActionDataPreprocessor
            from mmaction.registry import MODELS

            @MODELS.register_module()
            class LongVideoDataPreprocessor(ActionDataPreprocessor):
                """DataPreprocessor for 2D recognizer to infer on long video.

                Which would stack the num_clips to batch dimension, to preserve
                feature of each clip (no average among clips)
                """

                def __init__(self, num_frames=8, **kwargs) -> None:
                    super().__init__(**kwargs)
                    self.num_frames = num_frames

                def preprocess(self, inputs, data_samples, training=False):
                    batch_inputs, data_samples = super().preprocess(
                        inputs, data_samples, training)
                    # [N*M, T, C, H, W]
                    nclip_batch_inputs = batch_inputs.view(
                        (-1, self.num_frames) + batch_inputs.shape[2:])
                    # data_samples = data_samples * \
                    #     nclip_batch_inputs.shape[0]
                    return nclip_batch_inputs, data_samples

            preprocessor_cfg = cfg.model.data_preprocessor
            preprocessor_cfg.type = 'LongVideoDataPreprocessor'
            preprocessor_cfg['num_frames'] = clip_len

    # -------------------- Dump predictions --------------------
    args.dump = osp.join(args.output_prefix, 'total_feats.pkl')
    dump_metric = dict(type='DumpResults', out_file_path=args.dump)
    cfg.test_evaluator = [dump_metric]
    cfg.work_dir = osp.join(args.output_prefix, 'work_dir')

    return cfg


def filter_existing_videos(args):
    """Filter out videos that already have extracted features."""
    video_list = list_from_file(args.video_list)
    video_names = [line.split(' ')[0] for line in video_list]
    
    # Check which videos don't have feature files yet
    videos_to_process = []
    videos_with_features = []
    
    for i, video_name in enumerate(video_names):
        output_file = osp.join(args.output_prefix, video_name + '.pkl')
        if not osp.exists(output_file):
            videos_to_process.append(video_list[i])
        else:
            videos_with_features.append(video_name)
    
    config.logger.info(f"Found {len(videos_with_features)} videos with existing features, {len(videos_to_process)} videos to process")
    
    # If no videos need processing, return early
    if not videos_to_process:
        config.logger.info("All videos already have extracted features. Skipping feature extraction.")
        return False
    
    # Create a temporary filtered video list file if some videos need processing
    if len(videos_to_process) < len(video_list):
        temp_video_list = args.video_list + '.temp'
        with open(temp_video_list, 'w') as f:
            f.write('\n'.join(videos_to_process))
        args.video_list = temp_video_list
        config.logger.info(f"Created temporary video list with {len(videos_to_process)} videos to process")
    
    return True


def split_feats(args):
    config.logger.info("Splitting features and saving individual files.")
    total_feats = load(args.dump)
    
    # If you're dumping prediction scores instead of features
    if args.dump_score:
        total_feats = [sample['pred_scores']['item'] for sample in total_feats]

    video_list = list_from_file(args.video_list)
    video_list = [line.split(' ')[0] for line in video_list]

    # Track processing statistics
    processed_count = 0
    skipped_count = 0

    # Process each video sequentially
    for video_name, feature in zip(video_list, total_feats):
        output_file = osp.join(args.output_prefix, video_name + '.pkl')
        
        # Check if the feature file already exists
        if osp.exists(output_file):
            config.logger.debug(f"Feature file already exists for {video_name}, skipping.")
            skipped_count += 1
            continue
        
        config.logger.debug(f"Saving features for video: {video_name}")
        
        # Save the features for the current video
        dump(feature, output_file)
        processed_count += 1
        
        # Explicitly clear GPU memory after processing each video
        torch.cuda.empty_cache()  # This ensures the memory is freed after each video

    # After processing all videos, remove the total features dump file
    if osp.exists(args.dump):
        os.remove(args.dump)
    
    # Clean up temporary video list file if it was created
    if args.video_list.endswith('.temp'):
        os.remove(args.video_list)
        config.logger.info("Cleaned up temporary video list file")
    
    config.logger.info(f"Feature splitting completed. Processed: {processed_count}, Skipped: {skipped_count}")
    
def main():
    args = parse_args()

    # Check if we need to process any videos (filter out already processed ones)
    need_processing = filter_existing_videos(args)
    if not need_processing:
        config.logger.info("All videos already have extracted features. Exiting early.")
        return  # Exit early if all videos already have features

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher

    cfg.load_from = args.checkpoint

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()

    split_feats(args)


if __name__ == '__main__':
    main()
