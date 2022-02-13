import numpy as np
import torch
import torch.nn as nn

from srt.layers import RayEncoder, Transformer, PositionalEncoding
from srt.utils import nerf


class RayPredictor(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0, out_dims=3):
        super().__init__()
        self.query_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                        ray_octaves=15)
        self.transformer = Transformer(180, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=False)
        self.output_mlp = nn.Sequential(
            nn.Linear(180, 128),
            nn.ReLU(),
            nn.Linear(128, out_dims))

    def forward(self, z, x, rays):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """
        queries = self.query_encoder(x, rays)
        transformer_output = self.transformer(queries, z)
        output = self.output_mlp(transformer_output)
        return output


class SRTDecoder(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0):
        super().__init__()
        self.ray_predictor = RayPredictor(num_att_blocks=num_att_blocks,
                                          pos_start_octave=pos_start_octave,
                                          out_dims=3)

    def forward(self, z, x, rays, **kwargs):
        output = self.ray_predictor(z, x, rays)
        return torch.sigmoid(output), dict()


class NerfNet(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0, max_density=None):
        super().__init__()
        self.pos_encoder = PositionalEncoding(num_octaves=15, start_octave=pos_start_octave)
        self.ray_encoder = PositionalEncoding(num_octaves=15)

        self.transformer = Transformer(90, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=False)

        self.color_predictor = nn.Sequential(
            nn.Linear(179, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid())

        self.max_density = max_density

    def forward(self, z, x, rays):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query points [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """
        pos_enc = self.pos_encoder(x)
        h = self.transformer(pos_enc, z)

        density = h[..., 0]
        if self.max_density is not None and self.max_density > 0.:
            density = torch.sigmoid(density) * self.max_density
        else:
            density = nn.functional.relu(density)

        h = h[..., 1:]

        ray_enc = self.ray_encoder(rays)
        h = torch.cat((h, ray_enc), -1)
        colors = self.color_predictor(h)

        return density, colors


class NerfDecoder(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0, max_density=None, use_fine_net=True):
        super().__init__()
        nerf_kwargs = {'num_att_blocks': num_att_blocks,
                       'pos_start_octave': pos_start_octave,
                       'max_density': max_density}

        self.coarse_net = NerfNet(**nerf_kwargs)
        if use_fine_net:
            self.fine_net = NerfNet(**nerf_kwargs)
        else:
            self.fine_net = self.coarse_net

    def forward(self, z, x, rays, **render_kwargs):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """

        imgs, extras = render_nerf(self, z, x, rays, **render_kwargs)
        return imgs, extras

def eval_samples(scene_function, z, coords, rays):
    """
    Args:
        z: [batch_size, num_patches, c_dim]
        coords: [batch_size, num_samples, num_rays, 3]
        rays: [batch_size, num_rays, 3]
    Return:
        global_
    """
    batch_size, num_rays, num_samples = coords.shape[:3]

    rays_ext = rays.unsqueeze(1).repeat(1, 1, num_samples, 1).flatten(1, 2)

    density, color = scene_function(z, coords.flatten(1, 2), rays=rays_ext)

    density = density.view(batch_size, num_rays, num_samples)
    color = color.view(batch_size, num_rays, num_samples, 3)

    return density, color


def render_nerf(model, z, camera_pos, rays, num_coarse_samples=128,
                num_fine_samples=64, min_dist=0.035, max_dist=35., min_z=None,
                deterministic=False):
    """
    Render single NeRF image.
    Args:
        z: [batch_size, num_patches, c]
        camera_pos: camera position [batch_size, num_rays, 3]
        rays: [batch_size, num_rays, 3]
    """
    extras = {}

    coarse_depths, coarse_coords = nerf.get_nerf_sample_points(camera_pos, rays,
                                                               num_samples=num_coarse_samples,
                                                               min_dist=min_dist, max_dist=max_dist,
                                                               deterministic=deterministic,
                                                               min_z=min_z)

    coarse_densities, coarse_colors = eval_samples(model.coarse_net, z, coarse_coords, rays)

    coarse_img, coarse_depth, coarse_depth_dist = nerf.draw_nerf(
        coarse_densities, coarse_colors, coarse_depths)

    if num_fine_samples < 1:
        return coarse_img, {'depth': coarse_depth}
    elif num_fine_samples == 1:
        fine_depths = coarse_depth.unsqueeze(-1)
        fine_coords = camera_pos.unsqueeze(-2) + rays.unsqueeze(-2) * fine_depths.unsqueeze(-1)
    else:
        fine_depths, fine_coords = nerf.get_fine_nerf_sample_points(
            camera_pos, rays, coarse_depth_dist, coarse_depths,
            min_dist=min_dist, max_dist=max_dist, num_samples=num_fine_samples,
            deterministic=deterministic)

    fine_depths = fine_depths.detach()
    fine_coords = fine_coords.detach()

    depths_agg = torch.cat((coarse_depths, fine_depths), -1)
    coords_agg = torch.cat((coarse_coords, fine_coords), -2)

    depths_agg, sort_idxs = torch.sort(depths_agg, -1)
    coords_agg = torch.gather(coords_agg, -2, sort_idxs.unsqueeze(-1).expand_as(coords_agg))

    fine_pres, fine_values, = eval_samples(model.fine_net, z, coords_agg, rays)

    fine_img, fine_depth, depth_dist = nerf.draw_nerf(
        fine_pres, fine_values, depths_agg)

    def rgba_composite_white(rgba):
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:]
        result = torch.ones_like(rgb) * (1. - alpha) + rgb * alpha
        return result

    extras['depth'] = fine_depth
    extras['coarse_img'] = rgba_composite_white(coarse_img)
    extras['coarse_depth'] = coarse_depth

    return rgba_composite_white(fine_img), extras


