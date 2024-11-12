# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import cv2

data_file = '../../../data/quantex_share'
video_list = f'{data_file}/video_info_new.csv' # 
rawframe_dir = f'{data_file}/rawframes' # extracted rawframes
action_name_list = 'action_name.csv'
video_dir = f'{data_file}/videos'

train_rawframe_dir = rawframe_dir
val_rawframe_dir = rawframe_dir

json_file = f'{data_file}/quantex_share.json' #annotation file


def generate_rawframes_filelist():
    database = json.load(open(json_file))

    quantex_labels = open(action_name_list).readlines()
    quantex_labels = [x.strip() for x in quantex_labels[1:]]

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir)
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir)
    ]

    def simple_label(anno):
        label = anno[0]['label']
        return quantex_labels.index(label)

    def count_frames(dir_list, video):
        for dir_name in dir_list:
            if video in dir_name:
                return osp.basename(dir_name), len(os.listdir(dir_name))
        return None, None

    training = {}
    validation = {}
    key_dict = {}

    for k in database:
        data = database[k]
        subset = data['subset']

        if subset in ['training', 'validation']:
            annotations = data['annotations']
            label = simple_label(annotations)
            if subset == 'training':
                dir_list = train_dir_list
                data_dict = training
            else:
                dir_list = val_dir_list
                data_dict = validation

        else:
            continue

        gt_dir_name, num_frames = count_frames(dir_list, k)
        if gt_dir_name is None:
            continue
        data_dict[gt_dir_name] = [num_frames, label]
        key_dict[gt_dir_name] = k

    train_lines = [
        k + ' ' + str(training[k][0]) + ' ' + str(training[k][1])
        for k in training
    ]
    val_lines = [
        k + ' ' + str(validation[k][0]) + ' ' + str(validation[k][1])
        for k in validation
    ]

    with open(osp.join(data_file, 'quantex_share_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'quantex_share_val_video.txt'), 'w') as fout:
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
            label = quantex_labels.index(label)
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
        train_clips.extend(clip_list(k, database[key_dict[k]]))
    for k in validation:
        val_clips.extend(clip_list(k, database[key_dict[k]]))

    with open(osp.join(data_file, 'quantex_share_train_clip.txt'), 'w') as fout:
        fout.write('\n'.join(train_clips))
    with open(osp.join(data_file, 'quantex_share_val_clip.txt'), 'w') as fout:
        fout.write('\n'.join(val_clips))


if __name__ == '__main__':
    generate_rawframes_filelist()
