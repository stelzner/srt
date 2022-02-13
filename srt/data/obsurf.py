import numpy as np
import imageio
import yaml
from torch.utils.data import Dataset

import os

from srt.utils.nerf import get_camera_rays, get_extrinsic, transform_points


def downsample(x, num_steps=1):
    if num_steps is None or num_steps < 1:
        return x
    stride = 2**num_steps
    return x[stride//2::stride, stride//2::stride]


class Clevr3dDataset(Dataset):
    def __init__(self, path, mode, max_views=None, points_per_item=2048, canonical_view=True,
                 max_len=None, full_scale=False, max_objects=6, shapenet=False, downsample=None):
        """ Loads the dataset used in the ObSuRF paper.

        They may be downloaded at: https://stelzner.github.io/obsurf
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
            max_objects (int): Load only scenes with at most this many objects.
            shapenet (bool): Load ObSuRF's MultiShapeNet dataset, instead of CLEVR3D.
            downsample (int): Downsample height and width of input image by a factor of 2**downsample
        """
        self.path = path
        self.mode = mode
        self.points_per_item = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        self.full_scale = full_scale
        self.max_objects = max_objects
        self.shapenet = shapenet
        self.downsample = downsample

        self.max_num_entities = 5 if shapenet else 11
        self.num_views = 3

        if shapenet:
            self.start_idx, self.end_idx = {'train': (0, 80000),
                                            'val': (80000, 80500),
                                            'test': (90000, 100000)}[mode]
        else:
            self.start_idx, self.end_idx = {'train': (0, 70000),
                                            'val': (70000, 75000),
                                            'test': (85000, 100000)}[mode]

        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)
        num_available_views = self.metadata['camera_pos'].shape[1]

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_objects]

        dataset_name = 'ObSuRF-MSN' if shapenet else 'CLEVR3D'
        print(f'Initialized {dataset_name} {mode} set, {len(self.idxs)} examples')
        print(self.idxs)

        self.render_kwargs = {
            'min_dist': 0.035,
            'max_dist': 35.}

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs) * self.num_views

    def __getitem__(self, idx, noisy=True):
        scene_idx = idx % len(self.idxs)
        view_idx = idx // len(self.idxs)

        scene_idx = self.idxs[scene_idx]

        imgs = [np.asarray(imageio.imread(
            os.path.join(self.path, 'images', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]

        imgs = [img[..., :3].astype(np.float32) / 255 for img in imgs]

        input_image = downsample(imgs[view_idx], num_steps=self.downsample)

        input_images = np.expand_dims(np.transpose(input_image, (2, 0, 1)), 0)

        metadata = {k: v[scene_idx] for (k, v) in self.metadata.items()}

        all_rays = []
        all_camera_pos = metadata['camera_pos'][:self.num_views].astype(np.float32)
        for i in range(self.num_views):
            cur_rays = get_camera_rays(all_camera_pos[i], noisy=False)
            all_rays.append(cur_rays)
        all_rays = np.stack(all_rays, 0).astype(np.float32)

        input_camera_pos = all_camera_pos[view_idx]

        if self.canonical:
            track_point = np.zeros_like(input_camera_pos)  # All cameras are pointed at the origin
            canonical_extrinsic = get_extrinsic(input_camera_pos, track_point=track_point)
            canonical_extrinsic = canonical_extrinsic.astype(np.float32)
            all_rays = transform_points(all_rays, canonical_extrinsic, translate=False)
            all_camera_pos = transform_points(all_camera_pos, canonical_extrinsic)
            input_camera_pos = all_camera_pos[view_idx]

        input_rays = all_rays[view_idx]
        input_rays = downsample(input_rays, num_steps=self.downsample)
        input_rays = np.expand_dims(input_rays, 0)

        input_camera_pos = np.expand_dims(input_camera_pos, 0)

        all_pixels = np.reshape(np.stack(imgs, 0), (self.num_views * 240 * 320, 3))
        all_rays = np.reshape(all_rays, (self.num_views * 240 * 320, 3))
        all_camera_pos = np.tile(np.expand_dims(all_camera_pos, 1), (1, 240 * 320, 1))
        all_camera_pos = np.reshape(all_camera_pos, (self.num_views * 240 * 320, 3))

        num_points = all_rays.shape[0]

        if not self.full_scale:
            # If we have fewer points than we want, sample with replacement
            replace = num_points < self.points_per_item
            sampled_idxs = np.random.choice(np.arange(num_points),
                                            size=(self.points_per_item,),
                                            replace=replace)

            target_rays = all_rays[sampled_idxs]
            target_camera_pos = all_camera_pos[sampled_idxs]
            target_pixels = all_pixels[sampled_idxs]
        else:
            target_rays = all_rays
            target_camera_pos = all_camera_pos
            target_pixels = all_pixels

        result = {
            'input_images':         input_images,         # [1, 3, h, w]
            'input_camera_pos':     input_camera_pos,     # [1, 3]
            'input_rays':           input_rays,           # [1, h, w, 3]
            'target_pixels':        target_pixels,        # [p, 3]
            'target_camera_pos':    target_camera_pos,    # [p, 3]
            'target_rays':          target_rays,          # [p, 3]
            'sceneid':              idx,                  # int
        }

        if self.canonical:
            result['transform'] = canonical_extrinsic     # [3, 4] (optional)

        return result


