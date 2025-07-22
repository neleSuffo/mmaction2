import json
import csv
import config
import os
from typing import Dict, Tuple
from collections import defaultdict

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
    # Group video chunks by base video name
    grouped_annotations = defaultdict(list)
    for video_id, data in annotations.items():
        base_video_name = video_id.split('_')[0]  # Extract base name (e.g., "42000" from "42000_01")
        grouped_annotations[base_video_name].append((video_id, data))

    # Create a list of grouped videos with their total duration
    grouped_videos = [
        (base_name, sum(data['duration_second'] for _, data in chunks), chunks)
        for base_name, chunks in grouped_annotations.items()
    ]
    grouped_videos.sort(key=lambda x: x[1], reverse=True)  # Sort by total duration

    # Compute target durations for each split
    total_duration = sum(duration for _, duration, _ in grouped_videos)
    train_duration_target = total_duration * train_ratio
    test_val_ratio = (1 - train_ratio) / 2
    test_duration_target = total_duration * test_val_ratio
    val_duration_target = total_duration * test_val_ratio

    # Initialize splits
    training = {}
    testing = {}
    validation = {}

    train_duration = 0
    test_duration = 0
    val_duration = 0

    # Assign grouped videos to splits
    for _, group_duration, chunks in grouped_videos:
        if train_duration + group_duration <= train_duration_target:
            for video_id, data in chunks:
                training[video_id] = data
            train_duration += group_duration
        elif test_duration + group_duration <= test_duration_target:
            for video_id, data in chunks:
                testing[video_id] = data
            test_duration += group_duration
        else:
            for video_id, data in chunks:
                validation[video_id] = data
            val_duration += group_duration

    config.logger.info(
        f"Split videos into {len(training)} training (duration:{int(train_duration/60)} minutes), "
        f"{len(testing)} testing (duration:{int(test_duration/60)} minutes), and "
        f"{len(validation)} validation (duration:{int(val_duration/60)} minutes) sets"
    )
    return training, testing, validation

def generate_video_info_csv(annotations: Dict, output_csv: str) -> None:
    """
    Generates a CSV file with video information including video ID, number of frames, duration, fps, and subset.
    
    Parameters:
    ----------
    annotations : Dict
        A dictionary containing video annotations.
    output_csv : str
        The path to the output CSV file.

    Returns:
    -------
    None
        The function writes the video information to the specified CSV file.
    """
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