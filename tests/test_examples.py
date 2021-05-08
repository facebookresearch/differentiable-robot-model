import random

import pytest
import numpy as np
import torch

from examples import (
    learn_dynamics_iiwa,
    learn_forward_dynamics_iiwa,
    learn_kinematics_of_iiwa,
    learn_kinematics_of_toy,
)


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(0)


@pytest.mark.parametrize(
    "experiment",
    [
        learn_dynamics_iiwa,
        learn_forward_dynamics_iiwa,
        learn_kinematics_of_iiwa,
        learn_kinematics_of_toy,
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_examples(experiment, device):
    if not torch.cuda.is_available():
        device = "cpu"
    experiment.run(n_epochs=2, n_data=250, device=device)
