import os

import argparse
import copy
from collections import OrderedDict
from datetime import datetime
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils.logger import setup_logger
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything
from pixelflow.data_in1k import build_imagenet_loader


def get_args_parser():
    parser = argparse.ArgumentParser(description='Action Tokenizer', add_help=False)
    parser.add_argument('config', type=str, help='config')
    parser.add_argument('--output-dir', default='./exp0001', help='Output directory')
    parser.add_argument('--logging-steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--checkpoint-steps', type=int, default=1000, help='Checkpoint steps')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model')
    parser.add_argument('--report-to', type=str, default=None, help='Report to, eg. wandb')

    return parser


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    exp_name = "{}".format(datetime.now().strftime('%Y%m%d-%H%M%S'))

    # we print logs from all ranks to console, but only save to file from rank 0
    logger = setup_logger(args.output_dir, exp_name, rank, __name__)

    config = OmegaConf.load(args.config)

    rank_seed = config.seed * world_size + rank
    seed_everything(rank_seed, deterministic_ops=False, allow_tf32=False)
    logger.info(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size} Seed: {rank_seed}")

    # save args and config to output_dir
    with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
        f.write(str(args))
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))

    logger.info(f"Config: {config}")
    model = config_utils.instantiate_from_config(config.model).to(device)
    ema = copy.deepcopy(model).to(device)
    for param in ema.parameters():
        param.requires_grad = False

    logger.info(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    noise_scheduler = PixelFlowScheduler(
        num_train_timesteps=config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if args.pretrained_model is not None:
        ckpt = torch.load(args.pretrained_model, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    data_loader, sampler = build_imagenet_loader(config, noise_scheduler_copy)

    logger.info("***** Running training *****")
    global_step = 0
    first_epoch = 0

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    for epoch in range(first_epoch, config.train.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()
            target = batch["target_values"].to(device)
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                model_output = model(
                    hidden_states=batch["pixel_values"].to(device),
                    encoder_hidden_states=None,
                    class_labels=torch.tensor(batch["input_ids"], device=device),
                    timestep=batch["timesteps"].to(device),
                    latent_size=batch["batch_latent_size"].to(device),
                    pos_embed=batch["pos_embed"].to(device),
                    cu_seqlens_q=batch["cumsum_q_len"].to(device),
                    cu_seqlens_k=None,
                    seqlen_list_q=batch["seqlen_list_q"],
                    seqlen_list_k=None,
                )

            loss = (model_output.float() - target.float()) ** 2
            loss_split = torch.split(loss, batch["seqlen_list_q"], dim=0)
            loss_items = torch.stack([x.mean() for x in loss_split])
            if "padding_size" in batch and batch["padding_size"] is not None and batch["padding_size"] > 0:
                loss_items = loss_items[:-1]

            loss = loss_items.mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model.module)

            if global_step % args.logging_steps == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, Grad Norm: {grad_norm.item()}")

            if global_step % args.checkpoint_steps == 0 and global_step > 0:
                if rank == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args
                        },
                        os.path.join(args.output_dir, f"model_{global_step}.pt"))
                    logger.info(f"Model saved at step {global_step}")
                dist.barrier()

            global_step += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
