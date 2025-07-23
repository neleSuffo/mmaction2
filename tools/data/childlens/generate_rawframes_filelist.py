import json
import os
import os.path as osp
import cv2
import config
import csv

train_rawframe_dir = config.FrameExtraction.rawframes_processed_dir
val_rawframe_dir = config.FrameExtraction.rawframes_processed_dir

def load_video_info(video_info_file):
    config.logger.info(f"Loading video info from {video_info_file}")
    video_info = {}
    with open(video_info_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video']
            video_info[video_id] = row
    return video_info

def generate_rawframes_filelist():
    config.logger.info(f"Loading annotations from {config.AnnotationProcessing.split_annotation_path}")
    annotations = json.load(open(config.AnnotationProcessing.split_annotation_path))
    video_info = load_video_info(config.VideoProcessing.video_info_path)

    labels = open(config.AnnotationProcessing.activity_names_list).readlines()
    labels = [x.strip() for x in labels[1:]]

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir)
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir)
    ]

    def simple_label(annotation):
        label = annotation[0]['label']
        return labels.index(label)
    
    def count_frames(dir_list, video):
        for dir_name in dir_list:
            if video in osp.basename(dir_name):  # Ensure correct directory match
                # Filter only frame files based on expected file naming convention
                frame_files = [
                    f for f in os.listdir(dir_name)
                    if osp.isfile(osp.join(dir_name, f)) and f.startswith("img_") and f.endswith(".jpg")
                ]
                return osp.basename(dir_name), len(frame_files)
        return None, None

    training = {}
    validation = {}
    key_dict = {}

    for video_id in annotations:
        config.logging.info(f"Processing video {video_id}")
        data = annotations[video_id]
        subset = video_info[video_id]['subset']

        if subset in ['training', 'validation']:
            video_annotations = data['annotations']
            if not video_annotations:
                config.logger.warning(f"No annotations found for video {video_id}. Skipping video.")
                continue  # Skip the video if there are no annotations
            label = simple_label(video_annotations)
            if subset == 'training':
                dir_list = train_dir_list
                data_dict = training
            else:
                dir_list = val_dir_list
                data_dict = validation

            gt_dir_name, num_frames = count_frames(dir_list, video_id)
            if gt_dir_name is None:
                continue
            data_dict[gt_dir_name] = [num_frames, label]
            key_dict[gt_dir_name] = video_id

    train_lines = [
        k + ' ' + str(training[k][0]) + ' ' + str(training[k][1])
        for k in training
    ]
    val_lines = [
        k + ' ' + str(validation[k][0]) + ' ' + str(validation[k][1])
        for k in validation
    ]

    with open(config.FrameExtraction.train_video_txt_path, 'w') as fout:
        fout.write('\n'.join(train_lines))
        config.logging.info("Stored training video list at " + str(config.FrameExtraction.train_video_txt_path))
    with open(config.FrameExtraction.val_video_txt_path, 'w') as fout:
        fout.write('\n'.join(val_lines))
        config.logging.info("Stored validation video list at " + str(config.FrameExtraction.val_video_txt_path))
    
    def clip_list(k, anno, video_anno):
        duration = anno['duration_frame']
        num_frames = video_anno[0]
        fps = num_frames / duration
        annotations = anno['annotations']
        lines = []
        for annotation in annotations:
            segment = annotation['segment']
            label = annotation['label']
            label = labels.index(label)
            start, end = int(segment[0] * fps), int(segment[1] * fps)
            duration = end - start + 1
            if end > num_frames - 1:
                end = num_frames - 1
            if duration < 0:
                print(f'{k} {start} {duration} {label}')
                print("FPS: ", fps)
                assert duration > 0
            newline = f'{k} {start} {duration} {label}'
            lines.append(newline)
        return lines

    train_clips, val_clips = [], []
    for k in training:
        train_clips.extend(clip_list(k, annotations[key_dict[k]], training[k]))
    for k in validation:
        val_clips.extend(clip_list(k, annotations[key_dict[k]], validation[k]))

    with open(config.FrameExtraction.train_clip_txt_path, 'w') as fout:
        fout.write('\n'.join(train_clips))
        config.logging.info("Stored training clip list at " + str(config.FrameExtraction.train_clip_txt_path))
    with open(config.FrameExtraction.val_clip_txt_path, 'w') as fout:
        fout.write('\n'.join(val_clips))
        config.logging.info("Stored validation clip list at " + str(config.FrameExtraction.val_clip_txt_path))

if __name__ == '__main__':
    generate_rawframes_filelist()