# Copyright (c) OpenMMLab. All rights reserved.
"""This file processes the annotation files and generates proper annotation
files for localizers."""
import json
import config
import numpy as np
    
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        return data

# Load annotation files
annotation_database = load_json(config.AnnotationProcessing.split_annotation_path)

# load video info file with information about subset
video_records = np.loadtxt(config.VideoProcessing.video_info_path, dtype=str, delimiter=',', skiprows=1)

video_dict_train = {}
video_dict_val = {}
video_dict_test = {}
video_dict_full = {}

for _, video_record in enumerate(video_records):
    video_name = video_record[0]
    video_info = annotation_database[video_name]
    video_subset = video_record[5]
    video_info['fps'] = float(video_record[3])
    video_info['rfps'] = float(video_record[4])

    # Check if annotations are empty
    if not video_info['annotations']:
        config.logger.warning(f"No annotations found for video {video_name}. Skipping video.")
        continue
    
    video_dict_full[video_name] = video_info

    if video_subset == 'training':
        video_dict_train[video_name] = video_info
    elif video_subset == 'testing':
        video_dict_test[video_name] = video_info
    elif video_subset == 'validation':
        video_dict_val[video_name] = video_info

config.logger.info(f"Total videos processed: {len(video_records)}")
config.logger.info(f"Training videos: {len(video_dict_train)}")
config.logger.info(f"Validation videos: {len(video_dict_val)}")
config.logger.info(f"Testing videos: {len(video_dict_test)}")

output_dir = config.bmn_preprocessing_dir
with open(f'{output_dir}/anno_train.json', 'w') as result_file:
    json.dump(video_dict_train, result_file)

with open(f'{output_dir}/anno_val.json', 'w') as result_file:
    json.dump(video_dict_val, result_file)

with open(f'{output_dir}/anno_test.json', 'w') as result_file:
    json.dump(video_dict_test, result_file)

with open(f'{output_dir}/anno_full.json', 'w') as result_file:
    json.dump(video_dict_full, result_file)
