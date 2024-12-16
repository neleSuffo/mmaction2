import json
import os
import os.path as osp
import cv2
import csv

annotations_dir = '/home/nele_pauline_suffo/ProcessedData/annotations_superannotate'
json_file = f'{annotations_dir}/childlens_annotations.json'  # annotation file
video_info_file = f'{annotations_dir}/video_info.csv'  # video info file
action_name_list = f'{annotations_dir}/action_name.csv'

data_file = '/home/nele_pauline_suffo/ProcessedData/bmn_preprocessing'  # output directory
rawframe_dir = f'{data_file}/rawframes'  # extracted rawframes
video_dir = '/home/nele_pauline_suffo/ProcessedData/videos_superannotate_all'

train_rawframe_dir = rawframe_dir
val_rawframe_dir = rawframe_dir

def load_video_info(video_info_file):
    video_info = {}
    with open(video_info_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video'].replace('v_', '')
            video_info[video_id] = row
    return video_info

def generate_rawframes_filelist():
    annotations = json.load(open(json_file))
    video_info = load_video_info(video_info_file)

    childlens_labels = open(action_name_list).readlines()
    childlens_labels = [x.strip() for x in childlens_labels[1:]]

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir)
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir)
    ]

    def simple_label(anno):
        label = anno[0]['label']
        return childlens_labels.index(label)
    
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
        data = annotations[video_id]
        subset = video_info[video_id]['subset']

        if subset in ['training', 'validation']:
            video_annotations = data['annotations']
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

    with open(osp.join(data_file, 'childlens_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'childlens_val_video.txt'), 'w') as fout:
        fout.write('\n'.join(val_lines))

    def get_video_info(video_path):
        video = cv2.VideoCapture(video_path)
        
        # Get FPS and number of frames
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video.release()
        return fps, num_frames
    
    def clip_list(k, anno):
        fps, num_frames = get_video_info(osp.join(video_dir, k + '.MP4'))
        annotations = anno['annotations']
        lines = []
        for annotation in annotations:
            segment = annotation['segment']
            label = annotation['label']
            label = childlens_labels.index(label)
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
        train_clips.extend(clip_list(k, annotations[key_dict[k]]))
    for k in validation:
        val_clips.extend(clip_list(k, annotations[key_dict[k]]))

    with open(osp.join(data_file, 'childlens_train_clip.txt'), 'w') as fout:
        fout.write('\n'.join(train_clips))
    with open(osp.join(data_file, 'childlens_val_clip.txt'), 'w') as fout:
        fout.write('\n'.join(val_clips))

if __name__ == '__main__':
    generate_rawframes_filelist()