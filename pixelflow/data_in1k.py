# ImageNet-1K Dataset and DataLoader

from einops import rearrange
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import math
from functools import partial
import numpy as np
import random

from diffusers.models.embeddings import get_2d_rotary_pos_embed

# https://github.com/facebookresearch/DiT/blob/main/train.py#L85
def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def collate_fn(examples, config, noise_scheduler_copy):
    patch_size = config.model.params.patch_size
    pixel_values = torch.stack([eg[0] for eg in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [eg[1] for eg in examples]

    batch_size = len(examples)
    stage_indices = list(range(config.scheduler.num_stages)) * (batch_size // config.scheduler.num_stages + 1)
    stage_indices = stage_indices[:batch_size]

    random.shuffle(stage_indices)
    stage_indices = torch.tensor(stage_indices, dtype=torch.int32)
    orig_height, orig_width = pixel_values.shape[-2:]
    timesteps = torch.randint(0, config.scheduler.num_train_timesteps, (batch_size,))

    sample_list, input_ids_list, pos_embed_list, seq_len_list, target_list, timestep_list = [], [], [], [], [], []
    for stage_idx in range(config.scheduler.num_stages):
        corrected_stage_idx = config.scheduler.num_stages - stage_idx - 1
        stage_select_indices = timesteps[stage_indices == corrected_stage_idx]
        Timesteps = noise_scheduler_copy.Timesteps_per_stage[corrected_stage_idx][stage_select_indices].float()
        batch_size_select = Timesteps.shape[0]
        pixel_values_select = pixel_values[stage_indices == corrected_stage_idx]
        input_ids_select = [input_ids[i] for i in range(batch_size) if stage_indices[i] == corrected_stage_idx]

        end_height, end_width = orig_height // (2 ** stage_idx), orig_width // (2 ** stage_idx)

        ################ build model input ################
        start_t, end_t = noise_scheduler_copy.start_t[corrected_stage_idx], noise_scheduler_copy.end_t[corrected_stage_idx]

        pixel_values_end = pixel_values_select
        pixel_values_start = pixel_values_select
        if stage_idx > 0:
            # pixel_values_end
            for downsample_idx in range(1, stage_idx + 1):
                pixel_values_end = F.interpolate(pixel_values_end, (orig_height // (2 ** downsample_idx), orig_width // (2 ** downsample_idx)), mode="bilinear")

        # pixel_values_start
        for downsample_idx in range(1, stage_idx + 2):
            pixel_values_start = F.interpolate(pixel_values_start, (orig_height // (2 ** downsample_idx), orig_width // (2 ** downsample_idx)), mode="bilinear")
        # upsample pixel_values_start
        pixel_values_start = F.interpolate(pixel_values_start, (end_height, end_width), mode="nearest")

        noise = torch.randn_like(pixel_values_end)
        pixel_values_end = end_t * pixel_values_end + (1.0 - end_t) * noise
        pixel_values_start = start_t * pixel_values_start + (1.0 - start_t) * noise
        target = pixel_values_end - pixel_values_start

        t_select = noise_scheduler_copy.t_window_per_stage[corrected_stage_idx][stage_select_indices].flatten()
        while len(t_select.shape) < pixel_values_start.ndim:
            t_select = t_select.unsqueeze(-1)
        xt = t_select.float() * pixel_values_end + (1.0 - t_select.float()) * pixel_values_start
        
        target = rearrange(target, 'b c (h ph) (w pw) -> (b h w) (c ph pw)', ph=patch_size, pw=patch_size)
        xt = rearrange(xt, 'b c (h ph) (w pw) -> (b h w) (c ph pw)', ph=patch_size, pw=patch_size)

        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=config.model.params.attention_head_dim,
            crops_coords=((0, 0), (end_height // patch_size, end_width // patch_size)),
            grid_size=(end_height // patch_size, end_width // patch_size),
        )
        seq_len = (end_height // patch_size) * (end_width // patch_size)
        assert end_height == end_width, f"only support square image, got {seq_len}; TODO: latent_size_list"
        sample_list.append(xt)
        target_list.append(target)
        pos_embed_list.extend([pos_embed] * batch_size_select)
        seq_len_list.extend([seq_len] * batch_size_select)
        timestep_list.append(Timesteps)
        input_ids_list.extend(input_ids_select)

    pixel_values = torch.cat(sample_list, dim=0).to(memory_format=torch.contiguous_format)
    target_values = torch.cat(target_list, dim=0).to(memory_format=torch.contiguous_format)
    pos_embed = torch.cat([torch.stack(one_pos_emb, -1) for one_pos_emb in pos_embed_list], dim=0).float()
    cumsum_q_len = torch.cumsum(torch.tensor([0] + seq_len_list), 0).to(torch.int32)
    latent_size_list = torch.tensor([int(math.sqrt(seq_len)) for seq_len in seq_len_list], dtype=torch.int32)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids_list,
        "pos_embed": pos_embed,
        "cumsum_q_len": cumsum_q_len,
        "batch_latent_size": latent_size_list,
        "seqlen_list_q": seq_len_list,
        "cumsum_kv_len": None,
        "batch_kv_len": None,
        "timesteps": torch.cat(timestep_list, dim=0),
        "target_values": target_values,
    }


def build_imagenet_loader(config, noise_scheduler_copy):
    if config.data.center_crop:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.data.resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(round(config.data.resolution * config.data.expand_ratio), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(config.data.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = ImageFolder(config.data.root, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        seed=config.seed,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        collate_fn=partial(collate_fn, config=config, noise_scheduler_copy=noise_scheduler_copy),
        shuffle=False,
        sampler=sampler,
        num_workers=config.data.num_workers,
        drop_last=True,
    )
    return loader, sampler
