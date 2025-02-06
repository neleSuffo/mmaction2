# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config, DictAction
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer


# Defined variables
config_path = 'configs/localization/bmn/bmn_2xb8-400x100-9e_childlens-feature.py'
checkpoint_path = '/home/nele_pauline_suffo/outputs/bmn/best_auc_epoch_18.pth'
img_path = '/home/nele_pauline_suffo/ProcessedData/childlens_videos_processed/119281_01.mp4'
label_file = '/home/nele_pauline_suffo/ProcessedData/childlens_annotations/labels.txt'
out_filename = 'bmn_output_119281_01.mp4' 
device = 'cuda:0'
fps = 30
font_scale = None
font_color = 'white'
target_resolution = None  # Adjust if needed


# build the model from a config file and a checkpoint file
model = init_recognizer(config_path, checkpoint_path, device=device)

# test a single image (video in this case)
pred_result = inference_recognizer(model, img_path)

def get_output(
    video_path: str,
    out_filename: str,
    data_sample: str,
    labels: list,
    fps: int = 30,
    font_scale: Optional[str] = None,
    font_color: str = 'white',
    target_resolution: Optional[Tuple[int]] = None,
) -> None:
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path.
        out_filename (str): Output filename for the generated file.
        datasample (str): Predicted label of the generated file.
        labels (list): Label list of current dataset.
        fps (int): Number of picture frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the text. Defaults to None.
        font_color (str): Font color of the text. Defaults to ``white``.
        target_resolution (Tuple[int], optional): Set to
            (desired_width desired_height) to have resized frames. If
            either dimension is None, the frames are resized by keeping
            the existing aspect ratio. Defaults to None.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    # init visualizer
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)

    text_cfg = {'colors': font_color}
    if font_scale is not None:
        text_cfg.update({'font_sizes': font_scale})

    visualizer.add_datasample(
        out_filename,
        video_path,
        data_sample,
        draw_pred=True,
        draw_gt=False,
        text_cfg=text_cfg,
        fps=fps,
        out_type=out_type,
        out_path=osp.join('demo', out_filename),
        target_resolution=target_resolution)


def main():
    # Load config file
    cfg = Config.fromfile(config_path)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, checkpoint_path, device=device)
    pred_result = inference_recognizer(model, img_path)

    pred_scores = pred_result.pred_score.tolist()
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    # Read labels from file
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    results = [(labels[k[0]], k[1]) for k in top5_label]

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    if out_filename is not None:

        if target_resolution is not None:
            if target_resolution[0] == -1:
                assert isinstance(target_resolution[1], int)
                assert target_resolution[1] > 0
            if target_resolution[1] == -1:
                assert isinstance(target_resolution[0], int)
                assert target_resolution[0] > 0
            target_resolution = tuple(target_resolution)

        get_output(
            img_path,
            out_filename,
            pred_result,
            labels,
            fps=fps,
            font_scale=font_scale,
            font_color=font_color,
            target_resolution=target_resolution)


if __name__ == '__main__':
    main()