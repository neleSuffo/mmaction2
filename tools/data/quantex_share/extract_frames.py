import subprocess
import os
import config

def main():
    # Store the current directory for returning back later
    original_dir = os.getcwd()
    current_dir_name = os.path.basename(original_dir)  # Get the name of the current directory

    try:
        # Change to the parent directory
        os.chdir("..")
        
        # Build the command dynamically from config.py
        command = [
            "python", "build_rawframes.py",
            config.FrameExtraction.video_input_dir, config.FrameExtraction.rawframes_output_dir,
            "--level", str(config.FrameExtraction.level),
            "--flow-type", config.FrameExtraction.flow_type,
            "--ext", config.FrameExtraction.video_ext,
            "--task", config.FrameExtraction.task,
            "--new-short", str(config.FrameExtraction.new_short)
        ]
        
        # Add the --resume flag if enabled
        if config.FrameExtraction.resume:
            command.append("--resume")
        
        # Print the command for logging/debugging purposes
        print("Executing command:", " ".join(command))
        
        # Execute the command
        subprocess.run(command, check=True)
        
    finally:
        # Change directory back to original
        os.chdir(original_dir)
        
        # Change back to the directory with the name stored earlier
        target_dir = os.path.join(original_dir, current_dir_name)
        if os.path.exists(target_dir):
            os.chdir(target_dir)
            print(f"Changed back to directory: {target_dir}")
        else:
            print(f"Directory '{current_dir_name}' does not exist in {original_dir}")

if __name__ == "__main__":
    main()