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
    Memory-optimized version with file existence checks.
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
            
            # Skip video splitting if the output video already exists
            if not os.path.exists(new_video_path):
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
            else:
                logging.info(f"Video chunk {chunk_number} already exists, skipping video split")
            
            # Create a folder for the split rawframes
            split_rawframes_folder = os.path.join(rawframes_output_folder, new_video_base_name)
            os.makedirs(split_rawframes_folder, exist_ok=True)
            
            # Check if frames already exist in the split folder
            existing_frames = os.listdir(split_rawframes_folder) if os.path.exists(split_rawframes_folder) else []
            if len(existing_frames) > 0:
                logging.info(f"Frames for chunk {chunk_number} already exist ({len(existing_frames)} files), checking for missing frames")
            
            # Reset frame numbering for each chunk
            frame_counter = 1
            frames_copied = 0
            frames_skipped = 0
            
            # Copy the corresponding frames into the split folder (process one frame at a time)
            for j in range(i + 1, min(i + chunk_size + 1, total_frames + 1)):
                # Source frame paths
                frame_x = os.path.join(rawframes_folder, video_name, f"flow_x_{j:05d}.jpg")
                frame_y = os.path.join(rawframes_folder, video_name, f"flow_y_{j:05d}.jpg")
                frame_raw = os.path.join(rawframes_folder, video_name, f"img_{j:05d}.jpg")
                
                # Destination frame paths
                new_frame_x_name = f"flow_x_{frame_counter:05d}.jpg"
                new_frame_y_name = f"flow_y_{frame_counter:05d}.jpg"
                new_frame_raw_name = f"img_{frame_counter:05d}.jpg"
                
                dest_frame_x = os.path.join(split_rawframes_folder, new_frame_x_name)
                dest_frame_y = os.path.join(split_rawframes_folder, new_frame_y_name)
                dest_frame_raw = os.path.join(split_rawframes_folder, new_frame_raw_name)
                
                # Copy frames only if they don't already exist
                if os.path.exists(frame_x) and not os.path.exists(dest_frame_x):
                    shutil.copy2(frame_x, dest_frame_x)
                    frames_copied += 1
                elif os.path.exists(dest_frame_x):
                    frames_skipped += 1
                    
                if os.path.exists(frame_y) and not os.path.exists(dest_frame_y):
                    shutil.copy2(frame_y, dest_frame_y)
                    frames_copied += 1
                elif os.path.exists(dest_frame_y):
                    frames_skipped += 1
                    
                if os.path.exists(frame_raw) and not os.path.exists(dest_frame_raw):
                    shutil.copy2(frame_raw, dest_frame_raw)
                    frames_copied += 1
                elif os.path.exists(dest_frame_raw):
                    frames_skipped += 1
                
                frame_counter += 1
            
            logging.info(f"Chunk {chunk_number}: Copied {frames_copied} new frames, skipped {frames_skipped} existing frames")

        logging.info(f"Finished processing video: {video_name}")

if __name__ == "__main__":
    split_videos_and_frames(config.FrameExtraction.video_input_dir, config.FrameExtraction.rawframes_output_dir, config.FrameExtraction.videos_processed_dir, config.FrameExtraction.rawframes_processed_dir)