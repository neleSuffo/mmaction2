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


class ActivityLocalization:
    # directory with individual json annotations files
    annotations_dir = Path("/home/nele_pauline_suffo/ProcessedData/annotations_superannotate")
    # Path to file that contains all annotations combined (generated from generate_combined_annotations.py)
    combined_annotation_path = Path(f'{annotations_dir}/childlens_annotations.json')
    # List of labels to include in combined json file in "projects/mmaction2/tools/data/quantex_share/generate_combined_annotations.py"
    activities_to_include = [
    "Playing with Object",
    ]
    # train ratio for splitting videos into training and validation
    childlens_train_ratio: float = 0.8
    # Path to file that contains video information like number of frames, training and testing assignment(generated from generate_video_info.py)
    video_info_path = Path(f'{annotations_dir}/video_info.csv')
    bmn_preprocessing_dir = '/home/nele_pauline_suffo/ProcessedData/bmn_preprocessing'

class FrameExtraction:
    #parameters needed when running extract_frames.py
    video_input_dir = "/home/nele_pauline_suffo/ProcessedData/videos_superannotate_all"
    rawframes_output_dir = "/home/nele_pauline_suffo/ProcessedData/videos_superannotate/rawframes"
    # whether video files are stored as .mp4 or .MP4
    video_ext = "MP4"
    # extract only "rgb", "flow" or "both"
    task = "both"
    # keep as is
    new_short = 256
    level = 1
    flow_type = "tvl1"
    resume = True



rawframe_dir = "/home/nele_pauline_suffo/ProcessedData/bmn_preprocessing/rawframes"
video_dir = "/home/nele_pauline_suffo/ProcessedData/videos_superannotate_all"
action_name_list = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/action_name.csv"
json_file = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/childlens_annotations.json"
video_info_file = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/video_info.csv"





