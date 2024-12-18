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
    combined_annotation_path = Path("/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/childlens_annotations.json")
    # List of labels to include in combined json file in "projects/mmaction2/tools/data/quantex_share/generate_combined_annotations.py"
    activities_to_include = [
    "Playing with Object",
    ]

output_dir = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate"
data_file = "/home/nele_pauline_suffo/ProcessedData/bmn_preprocessing"
rawframe_dir = "/home/nele_pauline_suffo/ProcessedData/bmn_preprocessing/rawframes"
video_dir = "/home/nele_pauline_suffo/ProcessedData/videos_superannotate_all"
action_name_list = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/action_name.csv"
json_file = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/childlens_annotations.json"
video_info_file = "/home/nele_pauline_suffo/ProcessedData/annotations_superannotate/video_info.csv"





