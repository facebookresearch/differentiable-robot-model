# Copyright (c) Facebook, Inc. and its affiliates.
import torch


class UnconstrainedMassValue(torch.nn.Module):
    def __init__(self, init_val=None, device="cpu"):
        super(UnconstrainedMassValue, self).__init__()
        if init_val is None:
            self.mass = torch.nn.Parameter(torch.rand(1)).to(device)
        else:
            self.mass = torch.nn.Parameter(init_val).to(device)

    def forward(self):
        return self.mass


class PositiveMassValue(torch.nn.Module):
    def __init__(self, init_val=None, device="cpu"):
        super(PositiveMassValue, self).__init__()
        self.min_mass_val = torch.tensor(0.01).to(device)

        if init_val is None:
            init_param_value = torch.sqrt(torch.rand(1) ** 2)
        else:
            init_param_value = torch.sqrt(init_val)
        self.sqrt_mass = torch.nn.Parameter(init_param_value).to(device)

    def forward(self):
        return self.sqrt_mass * self.sqrt_mass + self.min_mass_val

#
