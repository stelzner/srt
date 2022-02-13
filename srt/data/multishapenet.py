from srt.utils.nerf import get_extrinsic, transform_points
from srt.utils.common import get_rank, get_world_size

from torch.utils.data import get_worker_info, IterableDataset
import numpy as np


class MultishapenetDataset(IterableDataset):
    def __init__(self, path, mode, points_per_item=8192, max_len=None, canonical_view=True,
                 full_scale=False):
        super(MultishapenetDataset).__init__()
        self.num_target_pixels = points_per_item
        self.path = path
        self.mode = mode
        self.canonical = canonical_view
        self.full_scale = full_scale

        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}

        import sunds  # Import here, so that only this dataset depends on Tensorflow
        builder = sunds.builder('multi_shapenet', data_dir=self.path)

        self.tf_dataset = builder.as_dataset(
            split=self.mode, 
            task=sunds.tasks.Nerf(yield_mode='stacked'),
        )

        self.num_items = 1000000 if mode == 'train' else 10000
        if max_len is not None:
            self.num_items = min(max_len, self.num_items)

        self.tf_dataset = self.tf_dataset.take(self.num_items)

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        dataset = self.tf_dataset

        if world_size > 1:
            num_shardable_items = (self.num_items // world_size) * world_size
            if num_shardable_items != self.num_items:
                print(f'MSN: Using {num_shardable_items} scenes to {self.mode} instead of {self.num_items} to be able to evenly shard to {world_size} processes.')
                dataset = dataset.take(num_shardable_items)
            dataset = dataset.shard(num_shards=world_size, index=rank)

        if self.mode == 'train':
            dataset = dataset.shuffle(1024)
        tf_iterator = dataset.as_numpy_iterator()

        for data in tf_iterator:
            yield self.prep_item(data)

    def prep_item(self, data):
        input_views = np.random.choice(np.arange(10), size=5, replace=False)
        target_views = np.array(list(set(range(10)) - set(input_views)))

        data['color_image'] = data['color_image'].astype(np.float32) / 255.

        input_images = np.transpose(data['color_image'][input_views], (0, 3, 1, 2))
        input_rays = data['ray_directions'][input_views]
        input_camera_pos = data['ray_origins'][input_views][:, 0, 0]

        if self.canonical:
            canonical_extrinsic = get_extrinsic(input_camera_pos[0], input_rays[0]).astype(np.float32)
            input_rays = transform_points(input_rays, canonical_extrinsic, translate=False)
            input_camera_pos = transform_points(input_camera_pos, canonical_extrinsic)

        target_pixels = np.reshape(data['color_image'][target_views], (-1, 3))
        target_rays = np.reshape(data['ray_directions'][target_views], (-1, 3))
        target_camera_pos = np.reshape(data['ray_origins'][target_views], (-1, 3))

        num_pixels = target_pixels.shape[0]

        if not self.full_scale:
            sampled_idxs = np.random.choice(np.arange(num_pixels),
                                            size=(self.num_target_pixels,),
                                            replace=False)

            target_pixels = target_pixels[sampled_idxs]
            target_rays = target_rays[sampled_idxs]
            target_camera_pos = target_camera_pos[sampled_idxs]

        if self.canonical:
            target_rays = transform_points(target_rays, canonical_extrinsic, translate=False)
            target_camera_pos = transform_points(target_camera_pos, canonical_extrinsic)

        sceneid = int(data['scene_name'][6:])

        result = {
            'input_images':         input_images,         # [k, 3, h, w]
            'input_camera_pos':     input_camera_pos,     # [k, 3]
            'input_rays':           input_rays,           # [k, h, w, 3]
            'target_pixels':        target_pixels,        # [p, 3]
            'target_camera_pos':    target_camera_pos,    # [p, 3]
            'target_rays':          target_rays,          # [p, 3]
            'sceneid':              sceneid,              # int
        }
        if self.canonical:
            result['transform'] = canonical_extrinsic     # [3, 4] (optional)
        return result

    def skip(self, n):
        """
        Skip the first n scenes
        """
        self.tf_dataset = self.tf_dataset.skip(n)




