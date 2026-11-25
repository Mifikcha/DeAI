import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from diloco import DiLoCo
from model import LanguageModel, ModelConfig


def check_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(10.0 + rank)
            model.bias.fill_(-10.0 - rank)
        outer = DiLoCo(model, lr=1.0, momentum=0.5)
        # Opposite local moves on weight and bias check both slices of the flat vector.
        for local_steps, expected_weight in [((2.0, 4.0), 14.5), ((0.0, 2.0), 16.75)]:
            with torch.no_grad():
                model.weight.add_(local_steps[rank])
                model.bias.sub_(local_steps[rank])
            outer.synchronize()
            torch.testing.assert_close(model.weight, torch.tensor([[expected_weight]]))
            torch.testing.assert_close(model.bias, torch.tensor([-expected_weight]))
    finally:
        dist.destroy_process_group()


class TrainingTests(unittest.TestCase):
    def test_two_rounds_match_hand_computed_nesterov(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = (Path(tmp) / "rendezvous").as_uri()
            mp.spawn(check_worker, args=(rendezvous,), nprocs=2)

    def test_prefix_cannot_see_future_tokens(self):
        model = LanguageModel(ModelConfig(context=8, width=16, heads=2, layers=1))
        x = torch.arange(8).unsqueeze(0)
        changed = x.clone()
        changed[:, 4:] += 100
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :4], model(changed)[:, :4])
