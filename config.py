#from dynaconf import Dynaconf
from collections import defaultdict
import logging
from pathlib import Path
from dynaconf import Dynaconf


settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml"],
)

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

data_dir = Path("/home/nele_pauline_suffo/ProcessedData")
dataset_name = "childlens"

class AnnotationProcessing:
    # directory with individual json annotations files
    annotations_dir = Path(f"{data_dir}/{dataset_name}_annotations")
    # Path to file that contains all annotations combined (generated from generate_combined_annotations.py)
    combined_annotation_path = Path(f"{annotations_dir}/{dataset_name}_annotations.json")
    # List of labels to include in combined json file in "projects/mmaction2/tools/data/quantex_share/generate_combined_annotations.py"
    activities_to_include = [
    "Playing with Object",
    "Drawing",
    "Reading a Book"
    ]
    # activities_to_include = [
    # "Playing with Object",
    # "Playing without Object",
    # "Watching Something",
    # "Drawing",
    # "Crafting Things",
    # "Dancing",
    # "Making Music",
    # "Pretend play",
    # "Reading a Book",
    # ]
    activity_names_list = f'{annotations_dir}/action_name.csv'
    
class VideoProcessing:
    # train ratio for splitting videos into training and validation
    train_test_ratio: float = 0.8
    # Path to file that contains video information like number of frames, training and testing assignment(generated from generate_video_info.py)
    video_info_path = Path(f"{AnnotationProcessing.annotations_dir}/video_info.csv")
    bmn_preprocessing_dir = Path(f"{data_dir}/bmn_preprocessing")

class FrameExtraction:
    #parameters needed when running extract_frames.py & generate_rawframes_filelist.py
    train_video_txt_path = Path(f"{VideoProcessing.bmn_preprocessing_dir}/{dataset_name}_train_video.txt")
    val_video_txt_path = Path(f"{VideoProcessing.bmn_preprocessing_dir}/{dataset_name}_val_video.txt")
    train_clip_txt_path = Path(f"{VideoProcessing.bmn_preprocessing_dir}/{dataset_name}_train_clip.txt")
    val_clip_txt_path = Path(f"{VideoProcessing.bmn_preprocessing_dir}/{dataset_name}_val_clip.txt")
    video_input_dir = Path(f"{data_dir}/videos_superannotate_all")
    rawframes_output_dir = Path(f"{data_dir}/videos_superannotate/rawframes")
    # whether video files are stored as .mp4 or .MP4
    video_ext = "MP4"
    # extract only "rgb", "flow" or "both"
    task = "both"
    # keep as is
    new_short = 256
    level = 1
    flow_type = "tvl1"
    resume = True
    denseflow_installation_path = "/home/nele_pauline_suffo/app/bin/denseflow"
    
class FeatureExtraction:
    rgb_config_file = "tsn_extract_rgb_feat_config.py"
    flow_config_file = "tsn_extract_flow_feat_config.py"
    pretrained_model_url = ("https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth")
    rgb_output_dir = Path(f"{VideoProcessing.bmn_preprocessing_dir}/features_rgb")
    flow_output_dir = Path(f"{VideoProcessing.bmn_preprocessing_dir}/features_flow")
    video_list = Path(f"{VideoProcessing.bmn_preprocessing_dir}/{dataset_name}_train_video.txt")
    # interval (in frames) between the centers of adjacent clips.
    clip_interval = 16
    #interval (in frames) between consecutive frames in a clip
    frame_interval = 2