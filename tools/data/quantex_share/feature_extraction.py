import config
import subprocess

def generate_rgb_command():
    """Generate the command for RGB feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.FeatureExtraction.rgb_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.rgb_output_dir} "
        f"--video-list {config.FeatureExtraction.video_list} "
        f"--video-root {config.FrameExtraction.rawframes_output_dir} "
        f"--long-video-mode "
        f"--clip-interval {config.FeatureExtraction.clip_interval} "
        f"--frame-interval {config.FeatureExtraction.frame_interval}"
    )
    return command

def generate_flow_command():
    """Generate the command for Flow feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.FeatureExtraction.flow_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.flow_output_dir} "
        f"--video-list {config.FeatureExtraction.video_list} "
        f"--video-root {config.FrameExtraction.rawframes_output_dir} "
        f"--long-video-mode "
        f"--clip-interval {config.FeatureExtraction.clip_interval} "
        f"--frame-interval {config.FeatureExtraction.frame_interval}"
    )
    return command

def execute_command(command):
    """Execute a shell command."""
    config.logging.info(f"Executing command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
        config.logging.info("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        config.logging.error(f"Command failed with error: {e}")

def main():
    # Generate commands
    rgb_command = generate_rgb_command()
    flow_command = generate_flow_command()

    # Execute commands
    config.logging.info("Starting RGB feature extraction.")
    execute_command(rgb_command)

    config.logging.info("Starting Flow feature extraction.")
    execute_command(flow_command)

if __name__ == "__main__":
    main()