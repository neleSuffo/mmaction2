import config
import logging

def generate_rgb_command():
    """Generate the command for RGB feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.pretrained_model_url.rgb_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.rgb_output_dir} "
        f"--video-list {config.FeatureExtraction.video_list} "
        f"--video-root {config.FrameExtraction.rawframes_output_dir}"
        f"--long-video-mode "
        f"--clip-interval {config.FeatureExtraction.clip_interval} "
        f"--frame-interval {config.FeatureExtraction.frame_interval}"
    )
    return command

def generate_flow_command():
    """Generate the command for Flow feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.pretrained_model_url.flow_config_file} "
        f"{config.FeatureExtraction.pretrained_model_url} {config.FeatureExtraction.flow_output_dir} "
        f"--video-list {config.FeatureExtraction.video_list} "
        f"--video-root {config.FrameExtraction.rawframes_output_dir} "
        f"--long-video-mode "
        f"--clip-interval {config.FeatureExtraction.clip_interval} "
        f"--frame-interval {config.FeatureExtraction.frame_interval}"
    )
    return command

def main():
    rgb_command = generate_rgb_command()
    flow_command = generate_flow_command()

    # Log or print commands for debugging
    logging.ino("Generated RGB Command:")
    print(rgb_command)
    print("\nGenerated Flow Command:")
    print(flow_command)

if __name__ == "__main__":
    main()