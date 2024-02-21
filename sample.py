import typer
import torch
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision.transforms.functional import to_pil_image

def main(
        sample_num: int,
        output_dir='samples',
        image_size=128,
        timesteps=1000,
    ):
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000    # number of steps
    )

    trainer = Trainer(
        diffusion,
        'dataset/solder_short',
        train_batch_size = 128,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = True              # whether to calculate fid during training
    )
    trainer.load(30)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sampled_images = diffusion.sample(batch_size = sample_num)
    for i, image_tensor in enumerate(sampled_images):
        image = to_pil_image(image_tensor)
        image.save(output / f'gen-{i:04d}.jpg')

if __name__ == "__main__":
    typer.run(main)

