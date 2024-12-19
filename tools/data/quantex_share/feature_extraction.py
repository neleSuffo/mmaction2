import config
import logging

def generate_rgb_command():
    """Generate the command for RGB feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.RGB.config_file} "
        f"{config.RGB.model_url} {config.RGB.output_dir} "
        f"--video-list {config.RGB.video_list} "
        f"--video-root {config.RGB.video_root}"
    )
    return command

def generate_flow_command():
    """Generate the command for Flow feature extraction."""
    command = (
        f"python ../../misc/clip_feature_extraction.py {config.Flow.config_file} "
        f"{config.Flow.model_url} {config.Flow.output_dir} "
        f"--video-list {config.Flow.video_list} "
        f"--video-root {config.Flow.video_root} > output.txt"
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