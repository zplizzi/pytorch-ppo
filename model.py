"""
Model definitions.

These are modified versions of the models from
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
"""
import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    """Helper to flatten a tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNBase(nn.Module):
    """CNN model."""
    def __init__(self, num_channels, num_outputs, dist, hidden_size=512):
        """Initializer.
            num_channels: the number of channels in the input images (eg 3
                for RGB images, or 12 for a stack of 4 RGB images).
            num_outputs: the dimension of the output distribution.
            dist: the output distribution (eg Discrete or Normal).
            hidden_size: the size of the final actor+critic linear layers
        """
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.actor_linear = init_(nn.Linear(hidden_size, num_outputs))
        self.dist = dist

    def forward(self, x):
        """x should have shape (batch_size, num_channels, 84, 84)."""
        x = self.main(x)
        value = self.critic_linear(x)
        action_logits = self.actor_linear(x)
        return value, self.dist(action_logits)


class MLPBase(nn.Module):
    """Basic multi-layer linear model."""
    def __init__(self, num_inputs, num_outputs, dist, hidden_size=64):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        init2_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init2_(nn.Linear(hidden_size, num_outputs)))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

        self.dist = dist

    def forward(self, x):
        value = self.critic(x)
        action_logits = self.actor(x)
        return value, self.dist(action_logits)


class Discrete(nn.Module):
    """A module that builds a Categorical distribution from logits."""
    def __init__(self, num_outputs):
        super().__init__()

    def forward(self, x):
        # Do softmax on the proper dimesion with either batched or non
        # batched inputs
        if len(x.shape) == 3:
            probs = nn.functional.softmax(x, dim=2)
        elif len(x.shape) == 2:
            probs = nn.functional.softmax(x, dim=1)
        else:
            print(x.shape)
            raise
        dist = torch.distributions.Categorical(probs=probs)
        return dist


class Normal(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    Standard deviations are learned parameters in this module.
    """
    def __init__(self, num_outputs):
        super().__init__()
        # initial variance is e^0 = 1
        self.stds = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        dist = torch.distributions.Normal(loc=x, scale=self.stds.exp())

        # By default we get the probability of sampling each dimension of the
        # distribution. The full probability is the product of these, or
        # the sum since we're working with log probabilities.
        # So overwrite the log_prob function to handle this for us
        dist.old_log_prob = dist.log_prob
        dist.log_prob = lambda x: dist.old_log_prob(x).sum(-1)

        return dist
