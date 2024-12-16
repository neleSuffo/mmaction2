#!/usr/bin/env bash
cd ../
python build_rawframes.py /home/nele_pauline_suffo/ProcessedData/videos_superannotate_all /home/nele_pauline_suffo/ProcessedData/videos_superannotate/rawframes --level 1 --flow-type tvl1 --ext MP4 --task both  --new-short 256
echo "Raw frames (RGB and tv-l1) Generated for train set"

cd quantex_share/
