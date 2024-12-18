import json
import csv
import config
import os
from typing import Dict, Tuple

def load_annotations(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def split_videos(annotations: Dict, train_ratio: float = config.VideoProcessing.childlens_train_ratio) -> Tuple[Dict, Dict]:
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
    
    config.logger.info(f"Split videos into {len(training)} training and {len(validation)} validation")
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
    config.logger.info(f"Saved video information to {output_csv}")


if __name__ == '__main__':
    annotations = load_annotations(config.AnnotationProcessing.combined_annotation_path)
    generate_video_info_csv(annotations, config.VideoProcessing.video_info_path)