import torch
import torch.optim as optim
import numpy as np
import imageio

import os, sys, argparse, math
import yaml, json
from tqdm import tqdm

from srt.data import get_dataset
from srt.checkpoint import Checkpoint
from srt.utils.visualize import visualize_2d_cluster, get_clustering_colors
from srt.utils.nerf import rotate_around_z_axis_torch, get_camera_rays, transform_points_torch, get_extrinsic_torch
from srt.model import SRT
from srt.trainer import SRTTrainer

from compile_video import compile_video_render


def get_camera_rays_render(camera_pos, **kwargs):
    rays = get_camera_rays(camera_pos[0], **kwargs)
    return np.expand_dims(rays, 0)

def lerp(x, y, t):
    return x + (y-x) * t

def easeout(t):
    return -0.5 * t**2 + 1.5 * t

def apply_fade(t, t_fade=0.2):
    v_max = 1. / (1. - t_fade)
    acc = v_max / t_fade
    if t <= t_fade:
        return 0.5 * acc * t**2
    pos_past_fade = 0.5 * acc * t_fade**2
    if t <= 1. - t_fade:
        return pos_past_fade + v_max * (t - t_fade)
    else:
        return 1. - 0.5 * acc * (t - 1.)**2

def get_camera_closeup(camera_pos, rays, t, zoomout=1., closeup=0.2, z_closeup=0.1, lookup=3.):
    orig_camera_pos = camera_pos[0] * zoomout
    orig_track_point = torch.zeros_like(orig_camera_pos)
    orig_ext = get_extrinsic_torch(orig_camera_pos, track_point=orig_track_point, fourxfour=True)

    final_camera_pos = closeup * orig_camera_pos
    final_camera_pos[2] = z_closeup * orig_camera_pos[2]
    final_track_point = orig_camera_pos + (orig_track_point - orig_camera_pos) * lookup
    final_track_point[2] = 0.

    cur_camera_pos = lerp(orig_camera_pos, final_camera_pos, t)
    cur_camera_pos[2] = lerp(orig_camera_pos[2], final_camera_pos[2], easeout(t))
    cur_track_point = lerp(orig_track_point, final_track_point, t)

    new_ext = get_extrinsic_torch(cur_camera_pos, track_point=cur_track_point, fourxfour=True)

    cur_rays = transform_points_torch(rays, torch.inverse(new_ext) @ orig_ext, translate=False)
    return cur_camera_pos.unsqueeze(0), cur_rays


def rotate_camera(camera_pos, rays, t):
    theta = math.pi * 2 * t
    camera_pos = rotate_around_z_axis_torch(camera_pos, theta)
    rays = rotate_around_z_axis_torch(rays, theta)

    return camera_pos, rays


def render3d(trainer, render_path, z, camera_pos, motion, transform=None, resolution=None, **render_kwargs):
    if transform is not None:  # Project camera into world space before applying motion transformations
        inv_transform = torch.inverse(transform)
        camera_pos = transform_points_torch(camera_pos, inv_transform)

    camera_pos_np = camera_pos.cpu().numpy()
    rays = torch.Tensor(get_camera_rays_render(camera_pos_np, **resolution)).to(camera_pos)

    for frame in tqdm(range(args.num_frames)):
        t = frame / args.num_frames
        if args.fade:
            t = apply_fade(t)
        if motion == 'rotate':  # Rotate camera around scene, tracking scene's center
            cur_camera_pos, cur_rays = rotate_camera(camera_pos, rays, t)
        elif motion == 'zoom':  # Stationary camera and track point, zoom in by reducing sensor width
            sensor_max = 0.032
            sensor_min = sensor_max / 5
            sensor_cur = lerp(sensor_max, sensor_min, frame / args.num_frames)
            cur_rays = get_camera_rays_render(camera_pos_np, sensor_width=sensor_cur, **resolution)
            cur_rays = torch.Tensor(cur_rays).float().cuda()
            cur_camera_pos = camera_pos
        elif motion == 'closeup':  # Move camera towards center of the scene, pan up slightly
            cur_camera_pos, cur_rays = get_camera_closeup(camera_pos, rays, t)
        elif motion == 'rotate_and_closeup':  # Rotate while moving in for a slight closeup
            t_closeup = ((-math.cos(t * math.pi * 2) + 1) * 0.5) * 0.5
            cur_camera_pos, cur_rays = get_camera_closeup(camera_pos, rays, t_closeup, lookup=1.5)
            cur_camera_pos, cur_rays = rotate_camera(cur_camera_pos, cur_rays, t)
        elif motion == 'eyeroll':  # Stationary camera, tracking circle around the scene
            theta = -t * 2 * math.pi
            track_point = 1.5 * np.array((math.cos(theta), math.sin(theta), 0))
            cur_rays = get_camera_rays_render(camera_pos_np, track_point=track_point, **resolution)
            cur_rays = torch.Tensor(cur_rays).float().cuda()
            cur_camera_pos = camera_pos
        else:
            raise ValueError(f'Unknown motion: {motion}')

        if transform is not None:  # Project camera back into canonical model coordinates
            cur_camera_pos = transform_points_torch(cur_camera_pos, transform)
            cur_rays = transform_points_torch(cur_rays, transform, translate=False)

        render, extras = trainer.render_image(z, cur_camera_pos, cur_rays, **render_kwargs)
        render = render.squeeze(0)
        render = render.cpu().numpy()
        render = (render * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(render_path, 'renders', f'{frame}.png'), render)

        if 'depth' in extras:
            depths = extras['depth'].squeeze(0).cpu().numpy()
            depths = (depths / render_kwargs['max_dist'] * 255.).astype(np.uint8)
            imageio.imwrite(os.path.join(render_path, 'depths', f'{frame}.png'), depths)


def process_scene(sceneid):
    render_path = os.path.join(out_dir, 'render', args.name, str(sceneid))
    if os.path.exists(render_path):
        print(f'Warning: Path {render_path} exists. Contents will be overwritten.')

    os.makedirs(render_path, exist_ok=True)
    subdirs = ['renders', 'depths']
    for d in subdirs:
        os.makedirs(os.path.join(render_path, d), exist_ok=True)

    if isinstance(val_dataset, torch.utils.data.IterableDataset):
        data = next(val_iterator)
    else:
        data = val_dataset.__getitem__(sceneid)

    input_images = torch.Tensor(data['input_images']).to(device).unsqueeze(0)
    input_camera_pos = torch.Tensor(data['input_camera_pos']).to(device).unsqueeze(0)
    input_rays = torch.Tensor(data['input_rays']).to(device).unsqueeze(0)

    resolution = {'height': input_rays.shape[2],
                  'width': input_rays.shape[3]}
    if args.height is not None:
        resolution['height'] = args.height
    if args.width is not None:
        resolution['width'] = args.width

    if 'transform' in data:
        transform = torch.Tensor(data['transform']).to(device)
    else:
        transform = None

    for i in range(input_images.shape[1]):
        input_np = (np.transpose(data['input_images'][i], (1, 2, 0)) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(render_path, f'input_{i}.png'), input_np)

    with torch.no_grad():
        z = model.encoder(input_images, input_camera_pos, input_rays)
                                          
        render3d(trainer, render_path, z, input_camera_pos[:, 0],
                 motion=args.motion, transform=transform, resolution=resolution, **render_kwargs)

    if not args.novideo:
        compile_video_render(render_path)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Render a video of a scene.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--num-frames', type=int, default=360, help='Number of frames to render.')
    parser.add_argument('--sceneid', type=int, default=0, help='Id of the scene to render.')
    parser.add_argument('--sceneid-start', type=int, help='Id of the scene to render.')
    parser.add_argument('--sceneid-stop', type=int, help='Id of the scene to render.')
    parser.add_argument('--height', type=int, help='Rendered image height in pixels. Defaults to input image height.')
    parser.add_argument('--width', type=int, help='Rendered image width in pixels. Defaults to input image width.')
    parser.add_argument('--name', type=str, help='Name of this sequence.')
    parser.add_argument('--motion', type=str, default='rotate', help='Type of sequence.')
    parser.add_argument('--sharpen', action='store_true', help='Square density values for sharper surfaces.')
    parser.add_argument('--parallel', action='store_true', help='Wrap model in DataParallel.')
    parser.add_argument('--train', action='store_true', help='Use training data.')
    parser.add_argument('--fade', action='store_true', help='Add fade in/out.')
    parser.add_argument('--it', type=int, help='Iteration of the model to load.')
    parser.add_argument('--render-kwargs', type=str, help='Renderer kwargs as JSON dict')
    parser.add_argument('--novideo', action='store_true', help="Don't compile rendered images into video")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
    print('configs loaded')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = os.path.dirname(args.config)
    exp_name = os.path.basename(out_dir)
    if args.name is None:
        args.name = args.motion
    if args.render_kwargs is not None:
        render_kwargs = json.loads(args.render_kwargs)
    else:
        render_kwargs = dict()

    model = SRT(cfg['model']).to(device)
    model.eval()

    mode = 'train' if args.train else 'val'
    val_dataset = get_dataset(mode, cfg['data'])

    render_kwargs |= val_dataset.render_kwargs

    optimizer = optim.Adam(model.parameters())
    trainer = SRTTrainer(model, optimizer, cfg, device, out_dir, val_dataset.render_kwargs)

    checkpoint = Checkpoint(out_dir, encoder=model.encoder, decoder=model.decoder, optimizer=optimizer)
    if args.it is not None:
        load_dict = checkpoint.load(f'model_{args.it}.pt')
    else:
        load_dict = checkpoint.load('model.pt')

    if args.sceneid_start is None:
        args.sceneid_start =  args.sceneid
        args.sceneid_stop = args.sceneid + 1

    if isinstance(val_dataset, torch.utils.data.IterableDataset):
        val_dataset.skip(args.sceneid_start)
        val_iterator = iter(val_dataset)

    for i in range(args.sceneid_start, args.sceneid_stop):
        process_scene(i)

