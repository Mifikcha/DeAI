import argparse
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from diloco import DiLoCo
from model import LanguageModel, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level Transformer with DiLoCo")
    parser.add_argument("--data", type=Path, default=Path("data/observations.txt"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo.pt"))
    parser.add_argument("--rounds", type=int, default=120, help="number of outer updates")
    parser.add_argument("--local-steps", type=int, default=20, help="AdamW steps per round")
    parser.add_argument("--batch-size", type=int, default=8, help="sequences per worker")
    parser.add_argument("--inner-lr", type=float, default=0.001)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--context", type=int, default=ModelConfig.context)
    parser.add_argument("--layers", type=int, default=ModelConfig.layers)
    parser.add_argument("--heads", type=int, default=ModelConfig.heads)
    parser.add_argument("--width", type=int, default=ModelConfig.width)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.rounds, args.local_steps, args.batch_size) < 1:
        parser.error("rounds, local-steps and batch-size must be positive")
    if not (args.inner_lr > 0 and args.outer_lr > 0 and 0 < args.momentum < 1):
        parser.error("use positive learning rates and momentum between 0 and 1")
    return args


def split_corpus(path, rank, world, context):
    size = path.stat().st_size
    split = size * 9 // 10
    if min(split // world, size - split) < context + 1:
        raise ValueError("Corpus too short for this context length and number of workers")
    data = torch.from_file(str(path), shared=False, size=size, dtype=torch.uint8)
    return data[split * rank // world : split * (rank + 1) // world], data[split:]


def sample_batch(data, batch_size, context, rng, device):
    starts = torch.randint(len(data) - context, (batch_size, 1), generator=rng)
    indices = starts + torch.arange(context + 1)
    tokens = data[indices].to(device=device, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


def next_byte_loss(model, inputs, targets):
    return F.cross_entropy(model(inputs).flatten(0, 1), targets.reshape(-1))


@torch.inference_mode()
def validation_loss(model, data, device):
    model.eval()
    rng = torch.Generator().manual_seed(31415)
    losses = []
    for _ in range(4):
        batch = sample_batch(data, 4, model.config.context, rng, device)
        losses.append(next_byte_loss(model, *batch))
    model.train()
    return torch.stack(losses).mean().item()


def save_model(path, model, rounds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        checkpoint = {
            "config": asdict(model.config),
            "model": model.state_dict(),
            "rounds": rounds,
        }
        torch.save(checkpoint, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def train(args, device, rank, world):
    config = ModelConfig(
        context=args.context, layers=args.layers, heads=args.heads, width=args.width
    )
    data, validation = split_corpus(args.data, rank, world, config.context)
    torch.manual_seed(args.seed)
    model = LanguageModel(config).to(device)
    outer = DiLoCo(model, args.outer_lr, args.momentum)
    inner = torch.optim.AdamW(model.parameters(), lr=args.inner_lr)
    rng = torch.Generator().manual_seed(args.seed + rank)
    if rank == 0:
        count = sum(param.numel() for param in model.parameters())
        print(f"workers={world} parameters={count} device={device} "
              f"initial_val_loss={validation_loss(model, validation, device):.4f}", flush=True)

    for round_number in range(1, args.rounds + 1):
        losses = torch.zeros((), device=device)
        for _ in range(args.local_steps):
            batch = sample_batch(data, args.batch_size, config.context, rng, device)
            inner.zero_grad(set_to_none=True)
            loss = next_byte_loss(model, *batch)
            loss.backward()
            inner.step()
            losses += loss.detach()
        outer.synchronize()
        if rank == 0:
            val = validation_loss(model, validation, device)
            print(f"round={round_number} train_loss_rank0={losses.item()/args.local_steps:.4f} "
                  f"val_loss={val:.4f}", flush=True)
    if rank == 0:
        save_model(args.output, model, args.rounds)
        print(f"saved={args.output}", flush=True)


def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device_type = args.device
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        train(args, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
