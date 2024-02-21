
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
from torchvision.transforms.functional import to_pil_image

import click
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from accelerate import Accelerator

#----------------------------------------------------------------------------

def gen_interp(G, output: str, seeds, shuffle_seed=None, w_frames=60*4, num_keyframes=None, device=torch.device('cuda')):
    if num_keyframes is None:
        num_keyframes = len(seeds)

    all_seeds = np.zeros(num_keyframes, dtype=np.int64)
    for idx in range(num_keyframes):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)
    
    image_size, channels = G.image_size, G.channels
    imgs = []
    output = Path(output)
    output.mkdir(exist_ok=True)
    for idx in range(len(all_seeds)-1):
        amounts = np.arange(0, 1, 1/num_keyframes)
        seed1 = all_seeds[idx]
        seed2 = all_seeds[idx+1]
        z1 = torch.from_numpy(np.random.RandomState(seed1).randn(1, channels, image_size, image_size).astype(np.float32)).to(device)
        z2 = torch.from_numpy(np.random.RandomState(seed2).randn(1, channels, image_size, image_size).astype(np.float32)).to(device)
        for alpha in amounts:
            img = G.interpolate(z1, z2, lam=alpha)[0]
            img = to_pil_image(G.unnormalize(img))
            name = output / f'seed{seed1}-{seed2}_{alpha:.2f}.jpg'
            imgs.append(img)
            img.save(name)
    return imgs

def gen_video(imgs, mp4: str, **video_kwargs):
    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for img in imgs:
        video_out.append_data(np.array(img))
    video_out.close()

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--model_path', help='model filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
def generate_images(
    model_path: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a image of interpolations for seeds 0 through 31.
    python interpolate.py --output=interpolate_samples --seeds=0-31 \\
        --model_path=results/model--30.pt

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % model_path)
    G, device = load_model(model_path)

    imgs = gen_interp(G=G, output=output, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, device=device)
    gen_video(imgs, mp4="intep.mp4", bitrate='12M')

#----------------------------------------------------------------------------


def load_model(model_path, split_batches=True, mixed_precision_type='fp16', amp=True):
    accelerator = Accelerator(
        split_batches = split_batches,
        mixed_precision = mixed_precision_type if amp else 'no'
    )

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )
    diffusion = CustomGaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000    # number of steps
    )

    device = accelerator.device
    data = torch.load(model_path, map_location=device)

    diffusion = accelerator.unwrap_model(diffusion)
    diffusion.load_state_dict(data['model'])
    diffusion = accelerator.prepare(diffusion)
    return diffusion, device
    
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

class CustomGaussianDiffusion(GaussianDiffusion):
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        if t is None:
            t = self.num_timesteps - 1

        assert x1.shape == x2.shape
        img = slerp(lam, x1, x2)
        img = self.ddim_sample(img)
        return img

    @torch.inference_mode()
    def ddim_sample(self, img):
        device, total_timesteps, sampling_timesteps, eta, objective = self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((1,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return img


if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
