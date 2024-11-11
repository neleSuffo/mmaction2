# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Optional, Union

import mmengine
from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset

from mmcv.transforms import BaseTransform
from mmcv.transforms import TRANSFORMS
import numpy as np

@DATASETS.register_module()
class ActivityNetDataset(BaseActionDataset):
    """ActivityNet dataset for temporal action localization. The dataset loads
    raw features and apply specified transforms to return a dict containing the
    frame tensors and other information. The ann_file is a json file with
    multiple objects, and each object has a key of the name of a video, and
    value of total frames of the video, total seconds of the video, annotations
    of a video, feature frames (frames covered by features) of the video, fps
    and rfps. Example of a annotation file:

    .. code-block:: JSON
        {
            "v_--1DO2V4K74":  {
                "duration_second": 211.53,
                "duration_frame": 6337,
                "annotations": [
                    {
                        "segment": [
                            30.025882995319815,
                            205.2318595943838
                        ],
                        "label": "Rock climbing"
                    }
                ],
                "feature_frame": 6336,
                "fps": 30.0,
                "rfps": 29.9579255898
            },
            "v_--6bJUbfpnQ": {
                "duration_second": 26.75,
                "duration_frame": 647,
                "annotations": [
                    {
                        "segment": [
                            2.578755070202808,
                            24.914101404056165
                        ],
                        "label": "Drinking beer"
                    }
                ],
                "feature_frame": 624,
                "fps": 24.0,
                "rfps": 24.1869158879
            },
            ...
        }
    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos are
            held. Defaults to ``dict(video='')``.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: Optional[ConfigType] = dict(video=''),
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        anno_database = mmengine.load(self.ann_file)
        for video_name in anno_database:
            video_info = anno_database[video_name]
            feature_path = video_name + '.csv'
            feature_path = '%s/%s' % (self.data_prefix['video'], feature_path)
            video_info['feature_path'] = feature_path
            #video_info['video_name'] = video_name
            video_info['filename'] = video_name
            data_list.append(video_info)
        return data_list

@DATASETS.register_module()
class CustomActivityNetDataset(ActivityNetDataset):
    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False):
        super().__init__(ann_file, pipeline, data_prefix, test_mode)
    
    def get_data_info(self, idx):
        data_info = super().get_data_info(idx)
        # Construct 'instances' key
        instances = []
        annotations = data_info.get('annotations', [])
        for annotation in annotations:
            instance = {}
            # Map 'label' to integer index using a label map
            instance['label'] = annotation['label']
            # Include the segment information
            instance['segment'] = annotation['segment']
            # Include other necessary instance-level annotations if needed
            instances.append(instance)
        data_info['instances'] = instances
        return data_info
    

@TRANSFORMS.register_module()
class LoadCustomAnnotations(BaseTransform):
    """Load and process the custom annotation format with labels and segments."""
    
    def __init__(self, with_label: bool = True, with_segment: bool = True, label_map: dict = None) -> None:
        """The constructor method of the class.

        Parameters
        ----------
        with_label : bool, optional
            whether to load label annotations, by default True
        with_segment : bool, optional
            whether to load segment annotations, by default True
        label_map : dict, optional
            a dictionary mapping the label strings to integer indices, by default None
        """
        super().__init__()
        self.with_label = with_label
        self.with_segment = with_segment

        self.label_map = label_map if label_map is not None else {
            "Playing with Object": 0,
            "Playing without Object": 1,
            "Pretend play": 2,
            "Watching Something": 3,
            "Reading a Book": 4,
            "Drawing": 5,
            "Crafting Things": 6,
            "Dancing": 7,
            "Making Music": 8,
            "Child Talking": 9,
            "Other Person Talking": 10,
            "Overheard Speech": 11,
            "Singing/Humming": 12,
            "Listening to Music/Audiobook": 13,
        }

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations."""
        gt_labels = []
        for instance in results['instances']:
            gt_labels.append(instance['label'])
        results['gt_labels'] = np.array(gt_labels, dtype=np.int64)

    def _load_segments(self, results: dict) -> None:
        """Private function to load segment annotations (start, end)."""
        gt_segments = []
        for instance in results['instances']:
            # Access the 'annotations' list and extract the segment
            for annotation in instance['annotations']:
                segment = annotation['segment']  # Extract segment (start, end)
                gt_segments.append(segment)
        # Store the result as an ndarray in (N, 2) format, where N is the number of segments
        results['gt_segments'] = np.array(gt_segments, dtype=np.float32)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types of annotations."""
        if self.with_label:
            self._load_labels(results)
        if self.with_segment:
            self._load_segments(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_label={self.with_label}, with_segment={self.with_segment})'
        return repr_str