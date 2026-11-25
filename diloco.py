import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils import parameters_to_vector


class DiLoCo:
    """One replica's shared weights and outer Nesterov optimizer."""

    def __init__(self, model, lr=0.7, momentum=0.9):
        self.local = list(model.parameters())
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.shared = nn.Parameter(parameters_to_vector(self.local).detach())
        if self.world > 1:
            with torch.no_grad():
                dist.broadcast(self.shared, src=0)
                self.copy_to_model()
        self.outer = torch.optim.SGD(
            [self.shared], lr=lr, momentum=momentum, nesterov=True
        )

    def parameter_views(self, vector):
        offset = 0
        for param in self.local:
            yield param, vector.narrow(0, offset, param.numel()).view_as(param)
            offset += param.numel()

    @torch.no_grad()
    def copy_to_model(self):
        for param, shared_view in self.parameter_views(self.shared):
            param.copy_(shared_view)

    @torch.no_grad()
    def synchronize(self):
        delta = self.shared.detach().clone()
        for param, delta_view in self.parameter_views(delta):
            delta_view.sub_(param)
        if self.world > 1:
            dist.all_reduce(delta, op=dist.ReduceOp.SUM)
            delta.div_(self.world)
        self.shared.grad = delta
        self.outer.step()
        self.outer.zero_grad(set_to_none=True)
        self.copy_to_model()
