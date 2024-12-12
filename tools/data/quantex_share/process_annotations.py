# Copyright (c) OpenMMLab. All rights reserved.
"""This file processes the annotation files and generates proper annotation
files for localizers."""
import json
import numpy as np

# labels for the boundary matching network (video only)
include_labels = ["Watching something",
                  "Drawing",
                  "Crafting things",
                  "Dancing",
                  "Playing with Object"]
    
def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


data_file = '../../../data/quantex_share'
info_file = f'{data_file}/video_info_new.csv'
ann_file = f'{data_file}/quantex_share_annotations_mmaction2.json'

anno_database = load_json(ann_file)

# Filter the annotations
filtered_annotations = {}
for video_id, video_data in anno_database.items():
    filtered_annotations[video_id] = {
        "duration_second": video_data["duration_second"],
        "duration_frame": video_data["duration_frame"],
        "annotations": [
            annotation for annotation in video_data["annotations"]
            if annotation["label"] in include_labels
        ]
    }

# Write the filtered annotations to a new JSON file
with open(f'{data_file}/filtered_quantex_share_annotations_mmaction2.json', 'w') as f:
    json.dump(filtered_annotations, f, indent=4)

ann_file_filtered = f'{data_file}/filtered_quantex_share_annotations_mmaction2.json'
anno_database_filtered = load_json(ann_file_filtered)

video_record = np.loadtxt(info_file, dtype=str, delimiter=',', skiprows=1)

video_dict_train = {}
video_dict_val = {}
video_dict_test = {}
video_dict_full = {}

for _, video_item in enumerate(video_record):
    video_name = video_item[0]
    video_info = anno_database_filtered[video_name]
    video_subset = video_item[5]
    video_info['fps'] = video_item[3].astype(np.float64)
    video_info['rfps'] = video_item[4].astype(np.float64)
    video_dict_full[video_name] = video_info
    if video_subset == 'training':
        video_dict_train[video_name] = video_info
    elif video_subset == 'testing':
        video_dict_test[video_name] = video_info
    elif video_subset == 'validation':
        video_dict_val[video_name] = video_info

print(f'full subset video numbers: {len(video_record)}')

with open(f'{data_file}/anno_train.json', 'w') as result_file:
    json.dump(video_dict_train, result_file)

with open(f'{data_file}/anno_val.json', 'w') as result_file:
    json.dump(video_dict_val, result_file)

with open(f'{data_file}/anno_test.json', 'w') as result_file:
    json.dump(video_dict_test, result_file)

with open(f'{data_file}/anno_full.json', 'w') as result_file:
    json.dump(video_dict_full, result_file)
