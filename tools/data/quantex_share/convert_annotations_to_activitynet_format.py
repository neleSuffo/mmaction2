import json
from pathlib import Path
from typing import Dict
import pandas as pd

# List of labels to include in the ActivityNet format
list_to_include = ['Playing with Object', 
                   'Playing without Object', 
                   'Pretend play',
                   'Watching Something',
                   'Reading a Book',
                   'Drawing',
                   'Crafting Things',
                   'Dancing',
                   'Making Music']


# Load the video_info_new.csv file
video_info_new_path = "/home/nele_pauline_suffo/projects/mmaction2/data/quantex_share/video_info_new.csv"
df_video_info_new = pd.read_csv(video_info_new_path)

# Function to get the subset for a given video ID
def get_subset(video_id):
    row = df_video_info_new[df_video_info_new['video'] == video_id]
    if not row.empty:
        return row.iloc[0]['subset']
    else:
        return None  # Video ID not found
    
# Function to read JSON from a file
def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Conversion function to ActivityNet format
def convert_annotations(data: Dict, fps: float = 30.0) -> Dict:
    # Initialize the converted structure
    converted_annotations = {}

    # Extract video ID, duration in seconds, and duration in frames
    video_id = data['metadata']['name']
    short_video_id = video_id.replace(".MP4", "")
    duration_microseconds = data['metadata']['duration']
    duration_seconds = duration_microseconds / 1_000_000.0
    # Extract the subset from the video_info_new.csv file
    subset = get_subset(int(short_video_id))

    # Initialize the video data structure in ActivityNet format
    converted_annotations[short_video_id] = {
        "subset": subset,
        "duration": duration_seconds,
        "url": "",  # Optional: Add video URL if available
        "annotations": []
    }

    # Loop through each annotation instance
    for item in data['instances']:
        meta = item['meta']
        if meta.get('className') in (None, 'Location'):
            continue

        # Extract start and end time
        start_time = meta["start"]
        end_time = meta["end"]

        # Process each parameter and add its first annotation to the list
        for parameter in item.get("parameters", []):
            timestamps = parameter.get("timestamps", [])

            # Check if there is at least one timestamp
            if timestamps and "attributes" in timestamps[0] and timestamps[0]["attributes"]:
                # Collect all "name" entries in a list
                names = [attr["name"] for timestamp in timestamps for attr in timestamp.get("attributes", [])]
                # Find the first name that is in the list_to_include
                label = next((name for name in names if name in list_to_include), None)
                
                # Add the annotation if a label was found
                if label is not None:
                    segment = [start_time / 1_000_000.0, end_time / 1_000_000.0]

                    # Append the annotation for this timestamp
                    converted_annotations[short_video_id]["annotations"].append({
                        "segment": segment,
                        "label": label
                    })

    return converted_annotations

# Function to process all JSON files in a folder and generate ActivityNet format
def process_all_json_files(input_dir: Path, output_file: Path, fps: float = 30.0) -> None:
    all_annotations = {
        "version": "1.0",
        "taxonomy": [
            {"nodeId": 1, "nodeName": "Playing with object", "parentId": None},
            {"nodeId": 2, "nodeName": "Playing without object", "parentId": None},
            {"nodeId": 3, "nodeName": "Pretend play", "parentId": None},
            {"nodeId": 4, "nodeName": "Watching something", "parentId": None},
            {"nodeId": 5, "nodeName": "Reading book", "parentId": None},
            {"nodeId": 6, "nodeName": "Drawing", "parentId": None},
            {"nodeId": 7, "nodeName": "Crafting things", "parentId": None},
            {"nodeId": 8, "nodeName": "Dancing", "parentId": None},
            {"nodeId": 9, "nodeName": "Making music", "parentId": None},
        ],
        "database": {}
    }

    # Iterate over all files in the specified folder
    for filename in input_dir.glob("*.json"):
        if filename.name == output_file.name:
            continue  # Skip the combined file
        # Read the JSON file
        data = read_json(filename)

        # Convert annotations and merge them into the "database" field
        video_annotations = convert_annotations(data, fps)
        all_annotations["database"].update(video_annotations)

    # Save combined_annotations as a JSON file in ActivityNet format
    with open(output_file, 'w') as file:
        json.dump(all_annotations, file, indent=4)

if __name__ == '__main__':
    input_dir = Path("/home/nele_pauline_suffo/ProcessedData/annotations_superannotate")
    output_file = Path("/home/nele_pauline_suffo/projects/mmaction2/data/quantex_share/quantex_share.json")
    process_all_json_files(input_dir, output_file)