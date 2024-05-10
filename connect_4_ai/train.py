import torch

from connect_4_ai.model_nn import create_model
from connect_4_ai.torch_utils import device


def train(step_count: int) -> None:
    torch.set_default_device(device())
    model = create_model()
    model.learn(total_timesteps=step_count)
    model.save("ppo_connect4")


if __name__ == "__main__":
    train(600)
