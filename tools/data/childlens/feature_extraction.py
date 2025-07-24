import config
import subprocess

def generate_rgb_command(video_list_path, split_name):
    """Generate the command for RGB feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.FeatureExtraction.rgb_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.rgb_output_dir} "
        f"--video-list {video_list_path} "
        f"--video-root {config.FrameExtraction.rawframes_processed_dir} "
    )
    return command

def generate_flow_command(video_list_path, split_name):
    """Generate the command for Flow feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.FeatureExtraction.flow_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.flow_output_dir} "
        f"--video-list {video_list_path} "
        f"--video-root {config.FrameExtraction.rawframes_processed_dir} "
    )
    return command

def execute_command(command):
    """Execute a shell command."""
    try:
        subprocess.run(command, shell=True, check=True)
        config.logging.info("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        config.logging.error(f"Command failed with error: {e}")

def main():
    # Define video list paths and split names
    video_lists = [
        (config.FrameExtraction.train_video_txt_path, "train"),
        (config.FrameExtraction.val_video_txt_path, "val"),
        (config.FrameExtraction.test_video_txt_path, "test")
    ]
    
    # Process each video list for both RGB and Flow features
    for video_list_path, split_name in video_lists:
        config.logging.info(f"Processing {split_name} dataset...")
        
        # Check if video list file exists
        if not osp.exists(video_list_path):
            config.logging.warning(f"Video list file not found: {video_list_path}. Skipping {split_name} dataset.")
            continue
        
        # RGB feature extraction
        config.logging.info(f"Starting RGB feature extraction for {split_name} dataset.")
        rgb_command = generate_rgb_command(video_list_path, split_name)
        execute_command(rgb_command)
        
        # Flow feature extraction
        config.logging.info(f"Starting Flow feature extraction for {split_name} dataset.")
        flow_command = generate_flow_command(video_list_path, split_name)
        execute_command(flow_command)
        
        config.logging.info(f"Completed feature extraction for {split_name} dataset.")
    
    config.logging.info("All feature extraction tasks completed.")

if __name__ == "__main__":
    main()