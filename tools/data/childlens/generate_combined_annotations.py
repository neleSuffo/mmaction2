import json
from pathlib import Path
from typing import Dict
import config
# Function to read JSON from a file
def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Conversion function
def convert_annotations(data: Dict, fps: float = 30.0) -> Dict:
    converted_annotations = {}
    
    # Extract video ID, duration in seconds, and duration in frames
    video_id = data['metadata']['name']
    short_video_id = video_id.replace(".MP4", "")
    duration_microseconds = data['metadata']['duration']
    duration_seconds = duration_microseconds / 1000000.0
    duration_frames = int(duration_seconds * fps)
    
    # Initialize the video data structure
    converted_annotations[short_video_id] = {
        "duration_second": duration_seconds, # duration in seconds
        "duration_frame": duration_frames,   # duration in frames
        "annotations": [],                   # initialize empty list for annotations
        "fps": fps,                          # frames per second
    }

    # Loop through each annotation instance
    for item in data['instances']:
        # Extract start and end time
        start_time = item["meta"]["start"]
        end_time = item["meta"]["end"]
        
        # Process each parameter and add its first annotation to the list
        for parameter in item.get("parameters", []):
            timestamps = parameter.get("timestamps", [])
            
            # Check if there is at least one timestamp
            if timestamps and "attributes" in timestamps[0] and timestamps[0]["attributes"]:
                # Collect all "name" entries in a list
                names = [attr["name"] for timestamp in timestamps for attr in timestamp.get("attributes", [])]
                # Find the first name that is in the list_to_include
                label = next((name for name in names if name in config.AnnotationProcessing.activities_to_include), None)
                # Add the annotation if a label was found
                if label is not None:
                    segment = [start_time / 1_000_000.0, end_time / 1_000_000.0]

                    # Append the annotation for this timestamp
                    converted_annotations[short_video_id]["annotations"].append({
                        "segment": segment,
                        "label": label
                    })
    
    return converted_annotations

# Function to process all JSON files in a folder
def process_all_json_files(folder_path: Path, 
                           output_file: Path,
                           fps: float = 30.0) -> Dict:
    all_annotations = {}
    
    # Find all JSON files in the specified folder
    json_files = list(folder_path.rglob("*.json"))
    num_files = len(json_files)
    config.logger.info(f"Found {num_files} JSON files in the folder {folder_path}")
    file_counter = 0
    
    # Iterate over all files in the specified folder
    for filename in json_files:
        if filename.name == output_file.name:
            continue  # Skip the combined file
        # Read the JSON file
        data = read_json(filename)
        
        # Convert annotations and merge them into the main dictionary (only if there are annotations)
        video_annotations = convert_annotations(data, fps)
        for _, annotations in video_annotations.items():
            if len(annotations["annotations"]) > 0:
                all_annotations.update(video_annotations)
                file_counter += 1
                    
    # Save combined_annotations as a JSON file
    with open(output_file, 'w') as file:
        json.dump(all_annotations, file, indent=4)
    config.logger.info(f"Saved {file_counter} combined annotations to {output_file}")

if __name__ == '__main__':
    process_all_json_files(config.AnnotationProcessing.annotations_dir, config.AnnotationProcessing.combined_annotation_path)