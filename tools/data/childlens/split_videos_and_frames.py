import os
import shutil
import cv2
import logging
import json
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def split_videos_and_frames(video_folder, rawframes_folder, output_folder, rawframes_output_folder, chunk_size=config.FrameExtraction.chunk_size):
    """
    Splits videos into chunks and copies associated rawframes into corresponding folders.
    """
    def load_video_ids(json_file):
        """Load video IDs from the JSON file."""
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        video_info = [(str(video_id) + f".{config.FrameExtraction.video_ext}", data['duration_frame']) for video_id, data in annotations.items()]
        return video_info

    video_info = load_video_ids(config.AnnotationProcessing.combined_annotation_path)
    
    for video_file, total_frames in video_info:
        video_name, _ = os.path.splitext(video_file)
        video_path = os.path.join(video_folder, video_file)

        # Get total frame count
        logging.info(f"Processing video: {video_name} with {total_frames} frames")

        # Split the video into chunks of specified size
        for i in range(0, total_frames, chunk_size):
            start_frame = i
            end_frame = min(i + chunk_size - 1, total_frames - 1)
            chunk_number = i // chunk_size + 1
            
            # Define the name and path of the new split video
            new_video_base_name = f"{video_name}_{chunk_number:02d}"
            new_video_name = f"{video_name}_{chunk_number:02d}.mp4"
            new_video_path = os.path.join(output_folder, new_video_name)
            
            # Create a folder for the split rawframes
            split_rawframes_folder = os.path.join(rawframes_output_folder, new_video_base_name)
            os.makedirs(split_rawframes_folder, exist_ok=True)
            config.logging.info(f"Created folder for split rawframes: {split_rawframes_folder}")

            # Construct the FFmpeg command
            command = (
                f"ffmpeg -i {video_path} "
                f"-vf \"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS\" "
                f"-an {new_video_path}"
            )
            logging.info(f"Executing FFmpeg command for chunk {chunk_number}")
            # Suppress detailed FFmpeg output
            with open(os.devnull, 'w') as devnull:
                os.system(f"{command} > /dev/null 2>&1")            
            # Reset frame numbering for each chunk
            frame_counter = 1
            
            # Copy the corresponding frames into the split folder
            for j in range(i + 1, i + chunk_size + 1):
                frame_x = os.path.join(rawframes_folder, video_name, f"flow_x_{j:05d}.jpg")
                frame_y = os.path.join(rawframes_folder, video_name, f"flow_y_{j:05d}.jpg")
                frame_raw = os.path.join(rawframes_folder, video_name, f"img_{j:05d}.jpg")
                
                new_frame_x_name = f"flow_x_{frame_counter:05d}.jpg"
                new_frame_y_name = f"flow_y_{frame_counter:05d}.jpg"
                new_frame_raw_name = f"img_{frame_counter:05d}.jpg"
                
                if os.path.exists(frame_x):
                    shutil.copy(frame_x, os.path.join(split_rawframes_folder, new_frame_x_name))
                if os.path.exists(frame_y):
                    shutil.copy(frame_y, os.path.join(split_rawframes_folder, new_frame_y_name))
                if os.path.exists(frame_raw):
                    shutil.copy(frame_raw, os.path.join(split_rawframes_folder, new_frame_raw_name))
                
                frame_counter += 1

        logging.info(f"Finished processing video: {video_name}")

if __name__ == "__main__":
    split_videos_and_frames(config.FrameExtraction.video_input_dir, config.FrameExtraction.rawframes_output_dir, config.FrameExtraction.videos_processed_dir, config.FrameExtraction.rawframes_processed_dir)