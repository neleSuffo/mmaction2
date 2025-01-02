import json
import csv
import config
import os
from typing import Dict, Tuple

def load_annotations(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def split_videos(annotations: Dict, train_ratio: float = config.VideoProcessing.train_test_ratio) -> Tuple[Dict, Dict, Dict]:
    """
    Splits the videos into training, testing, and validation sets.

    Args:
        annotations (Dict): A dictionary of video annotations.
        train_ratio (float): The ratio of videos to include in the training set.

    Returns:
        Tuple[Dict, Dict, Dict]: Dictionaries for training, testing, and validation sets.
    """
    test_val_ratio = (1 - train_ratio) / 2  # Split the remaining ratio evenly between testing and validation

    video_list = [(video_id, data['duration_second']) for video_id, data in annotations.items()]
    video_list.sort(key=lambda x: x[1], reverse=True)  # Sort by duration

    total_duration = sum(duration for _, duration in video_list)
    train_duration_target = total_duration * train_ratio
    test_duration_target = total_duration * test_val_ratio
    val_duration_target = total_duration * test_val_ratio

    training = {}
    testing = {}
    validation = {}

    train_duration = 0
    test_duration = 0
    val_duration = 0

    for video_id, duration in video_list:
        if len(training) / (len(training) + len(testing) + len(validation) + 1) < train_ratio and train_duration + duration <= train_duration_target:
            training[video_id] = annotations[video_id]
            train_duration += duration
        elif len(testing) / (len(training) + len(testing) + len(validation) + 1) < test_val_ratio and test_duration + duration <= test_duration_target:
            testing[video_id] = annotations[video_id]
            test_duration += duration
        else:
            validation[video_id] = annotations[video_id]
            val_duration += duration

    config.logger.info(
        f"Split videos into {len(training)} training (duration:{int(train_duration/60)} minutes), {len(testing)} testing (duration:{int(test_duration/60)} minutes), and {len(validation)} validation (duration:{int(val_duration/60)} minutes) sets"
    )
    return training, testing, validation

def generate_video_info_csv(annotations: Dict, output_csv: str):
    training, testing, validation = split_videos(annotations)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['video', 'numFrame', 'seconds', 'fps', 'rfps', 'subset', 'featureFrame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subset, data_dict in [('training', training), ('testing', testing), ('validation', validation)]:
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
    annotations = load_annotations(config.AnnotationProcessing.split_annotation_path)
    generate_video_info_csv(annotations, config.VideoProcessing.video_info_path)