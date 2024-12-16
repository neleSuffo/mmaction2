import json
import csv
import os
from typing import Dict, Tuple

def load_annotations(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def split_videos(annotations: Dict, train_ratio: float = 0.8) -> Tuple[Dict, Dict]:
    video_list = [(video_id, data['duration_second']) for video_id, data in annotations.items()]
    video_list.sort(key=lambda x: x[1], reverse=True)  # Sort by duration

    total_duration = sum(duration for _, duration in video_list)
    train_duration_target = total_duration * train_ratio

    training = {}
    validation = {}
    train_duration = 0
    val_duration = 0

    for video_id, duration in video_list:
        if len(training) / (len(training) + len(validation) + 1) < train_ratio and train_duration + duration <= train_duration_target:
            training[video_id] = annotations[video_id]
            train_duration += duration
        else:
            validation[video_id] = annotations[video_id]
            val_duration += duration

    return training, validation

def generate_video_info_csv(annotations: Dict, output_csv: str):
    training, validation = split_videos(annotations)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['video', 'numFrame', 'seconds', 'fps', 'rfps', 'subset', 'featureFrame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subset, data_dict in [('training', training), ('validation', validation)]:
            for video_id, data in data_dict.items():
                video_name = video_id
                num_frames = data['duration_frame']
                duration_seconds = data['duration_second']
                fps = data['fps']
                rfps = fps  # Assuming rfps is the same as fps
                feature_frame = num_frames  # Assuming featureFrame is the same as numFrame

                writer.writerow({
                    'video': video_name,
                    'numFrame': num_frames,
                    'seconds': duration_seconds,
                    'fps': fps,
                    'rfps': rfps,
                    'subset': subset,
                    'featureFrame': feature_frame
                })

if __name__ == '__main__':
    annotations_dir = '/home/nele_pauline_suffo/ProcessedData/annotations_superannotate'
    json_file = f'{annotations_dir}/childlens_annotations.json'  # annotation file
    output_csv = f'{annotations_dir}/video_info.csv'

    annotations = load_annotations(json_file)
    generate_video_info_csv(annotations, output_csv)