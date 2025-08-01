import json
import cv2
from pathlib import Path
from typing import Dict
import config

# Function to read JSON from a file
def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_frame_count(video_path: str) -> int:
    """
    Get the total number of frames in a video file.
    
    Parameters:
    ----------
    video_path : str
        The path to the video file.
    
    Returns:
    -------
    int
        The total number of frames in the video.
    """
    # Open the vadeo file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        raise ValueError(f"Could not open the video file: {video_path}")
    
    # Get the frame count
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video file
    video.release()
    
    return frame_count

def convert_annotations(data: Dict, fps: float = config.FrameExtraction.fps) -> Dict:
    """
    Convert annotations from the original format json format to the desired format.
    
    Parameters:
    ----------
    data : Dict
        The original annotations data loaded from a JSON file.
    fps : float
        Frames per second for the video processing.
        Default is set in FrameExtraction class.
        This is used to calculate the duration in seconds from frame counts.    
    
    Returns:
    -------
    Dict
        A dictionary containing the converted annotations with video IDs, durations, and activity segments.
    
    """
    converted_annotations = {}
    
    # Extract video ID, duration in seconds, and duration in frames
    video_id = data['video_name']
    short_video_id = video_id.replace(".MP4", "")
    duration_frames = get_frame_count(config.FrameExtraction.video_input_dir / video_id)
    duration_seconds = int(duration_frames / fps)
    
    # Initialize the video data structure
    converted_annotations[short_video_id] = {
        "duration_second": duration_seconds,
        "duration_frame": duration_frames,
        "annotations": [],
        "fps": fps,
    }

    # Loop through each annotation in the annotations array
    for annotation in data.get('annotations', []):
        # Get the event ID as the label
        label = annotation.get('eventId')
        
        # Only process annotations that are in our activities to include
        if label in config.AnnotationProcessing.activities_to_include:
            start_time = annotation.get('startTime', 0)
            end_time = annotation.get('endTime', 0)
            
            # Create segment (times are already in seconds)
            segment = [start_time, end_time]
            
            converted_annotations[short_video_id]["annotations"].append({
                "segment": segment,
                "label": label
            })
    
    return converted_annotations

def adjust_for_split_videos(data: Dict, chunk_size: int, fps: float = config.FrameExtraction.fps) -> Dict:
    """
    Adjust annotations for videos that are split into chunks.
    
    Parameters:
    ----------
    data : Dict
        The original annotations data loaded from a JSON file.
    chunk_size : int
        The number of frames in each chunk for splitting videos.
    fps : float
        Frames per second for the video processing.
        
    Returns:
    -------
    Dict
        A dictionary containing the split annotations with video IDs, durations, and activity segments.
    """
    split_annotations = {}
    chunks_in_video = 0  # Initialize a counter for chunks
    for video_id, video_data in data.items():
        total_frames = video_data["duration_frame"]
        annotations = video_data["annotations"]

        chunk_counter = 1
        for start_frame in range(0, total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size - 1, total_frames - 1)
            split_video_id = f"{video_id}_{chunk_counter:02d}"
            split_duration_seconds = (end_frame - start_frame + 1) / fps
            
            split_annotations[split_video_id] = {
                "duration_second": split_duration_seconds,
                "duration_frame": end_frame - start_frame + 1,
                "annotations": [],
                "fps": fps,
            }

            for annotation in annotations:
                annotation_start_frame = int(annotation["segment"][0] * fps)
                annotation_end_frame = int(annotation["segment"][1] * fps)
                
                # Check if the annotation overlaps with the current chunk
                if annotation_start_frame <= end_frame and annotation_end_frame >= start_frame:
                    # Adjust the segment to fit within the chunk's frame range
                    adjusted_segment = [
                        max(0, annotation_start_frame - start_frame) / fps,
                        min(annotation_end_frame - start_frame, chunk_size - 1) / fps,
                    ]
                    split_annotations[split_video_id]["annotations"].append({
                        "segment": adjusted_segment,
                        "label": annotation["label"]
                    })
            
            chunks_in_video += 1
            chunk_counter += 1
    
    return split_annotations, chunks_in_video

def process_all_json_files(annotations_dir: Path, 
                           output_file: Path,
                           split_output_file: Path,
                           chunk_size: int = config.FrameExtraction.chunk_size,
                           fps: float = config.FrameExtraction.fps) -> None:
    """
    Process all JSON files in the specified folder and generate combined and split annotations.
    
    Parameters:
    ----------
    annotations_dir : Path
        The path to the folder containing JSON files.
    output_file : Path
        The path to save the combined annotations JSON file.
    split_output_file : Path
        The path to save the split annotations JSON file.
    chunk_size : int
        The number of frames in each chunk for splitting videos.
    fps : float
        Frames per second for the video processing.
    """
    all_annotations = {}
    all_split_annotations = {}
    total_chunks = 0  # Total number of chunks across all videos

    json_files = list(annotations_dir.rglob("*.json"))
    num_files = len(json_files)
    config.logger.info(f"Found {num_files} JSON files in the folder {annotations_dir}")
    file_counter = 0
    
    for filename in json_files:
        if filename.name == config.AnnotationProcessing.combined_annotation_path.name or filename.name == config.AnnotationProcessing.split_annotation_path.name:
            continue
        data = read_json(filename)
        video_annotations = convert_annotations(data, fps)
        for video_id, annotations in video_annotations.items():
            if len(annotations["annotations"]) > 0:
                all_annotations.update(video_annotations)
                split_annotations, chunks_in_video = adjust_for_split_videos(video_annotations, chunk_size, fps)
                all_split_annotations.update(split_annotations)
                total_chunks += chunks_in_video
                file_counter += 1

    with open(output_file, 'w') as file:
        json.dump(all_annotations, file, indent=4)
    config.logger.info(f"Saved {file_counter} combined annotations to {output_file}")
    
    with open(split_output_file, 'w') as file:
        json.dump(all_split_annotations, file, indent=4)
    config.logger.info(f"Saved {file_counter} split annotations to {split_output_file}")
    config.logger.info(f"Total number of chunks generated across all videos: {total_chunks}")

if __name__ == '__main__':
    process_all_json_files(
        config.AnnotationProcessing.annotations_dir,
        config.AnnotationProcessing.combined_annotation_path,
        config.AnnotationProcessing.split_annotation_path
    )