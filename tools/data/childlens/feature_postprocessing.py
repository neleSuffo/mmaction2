import os
import os.path as osp
import multiprocessing
import numpy as np
import scipy.interpolate
from mmengine import dump, load
import config


def pool_feature(data, num_proposals=100, num_sample_bins=3, pool_type='mean'):
    """Pool features with arbitrary temporal length.

    Args:
        data (list[np.ndarray] | np.ndarray): Features of an untrimmed video,
            with arbitrary temporal length.
        num_proposals (int): The temporal dim of pooled feature. Default: 100.
        num_sample_bins (int): How many points to sample to get the feature
            vector at one timestamp. Default: 3.
        pool_type (str): Type of pooling to pool features. Choices are
            ['mean', 'max']. Default: 'mean'.

    Returns:
        np.ndarray: The pooled feature with shape num_proposals x feature_dim.
    """
    if len(data) == 1:
        return np.concatenate([data] * num_proposals)
    x_range = list(range(len(data)))
    f = scipy.interpolate.interp1d(x_range, data, axis=0)
    eps = 1e-4
    start, end = eps, len(data) - 1 - eps
    anchor_size = (end - start) / num_proposals
    ptr = start
    feature = []
    for _ in range(num_proposals):
        x_new = [
            ptr + i / num_sample_bins * anchor_size
            for i in range(num_sample_bins)
        ]
        y_new = f(x_new)
        if pool_type == 'mean':
            y_new = np.mean(y_new, axis=0)
        elif pool_type == 'max':
            y_new = np.max(y_new, axis=0)
        else:
            raise NotImplementedError('Unsupported pool type')
        feature.append(y_new)
        ptr += anchor_size
    feature = np.stack(feature)
    return feature


def merge_feat(name, output_dir, output_format, rgb_folder, flow_folder):
    # Concatenate RGB and flow features for a single sample
    rgb_feat = load(osp.join(rgb_folder, name))
    flow_feat = load(osp.join(flow_folder, name))
    rgb_feat = pool_feature(rgb_feat)
    flow_feat = pool_feature(flow_feat)
    feat = np.stack((rgb_feat, flow_feat), axis=-1)
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if output_format == 'pkl':
        dump(feat, osp.join(output_dir, name))
    elif output_format == 'csv':
        feat = feat.tolist()
        lines = []
        line0 = ','.join([f'f{i}' for i in range(400)])
        lines.append(line0)
        for line in feat:
            lines.append(','.join([f'{x:.4f}' for x in line]))
        with open(osp.join(output_dir, name.replace('.pkl', '.csv')), 'w') as f:
            f.write('\n'.join(lines))


def main():
    rgb_folder = config.FeatureExtraction.rgb_output_dir
    flow_folder = config.FeatureExtraction.flow_output_dir
    output_dir = config.FeatureExtraction.combined_feature_dir
    output_format = config.FeatureExtraction.feature_output_format
    
    rgb_feat = [file for file in os.listdir(rgb_folder) if file.endswith('.pkl')]
    flow_feat = [file for file in os.listdir(flow_folder) if file.endswith('.pkl')]
    
    assert set(rgb_feat) == set(flow_feat), "Mismatch between RGB and flow features."
    
    pool = multiprocessing.Pool(32)
    pool.starmap(merge_feat, [(file, output_dir, output_format, rgb_folder, flow_folder) for file in rgb_feat])

if __name__ == '__main__':
    main()