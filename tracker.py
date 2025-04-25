import datetime
import os
import time

import torch
import wandb

class WandBTracker:
    def __init__(self, name=None, args=None):
        wandb.init(project="pytorch-ppo", config=args)

    def add_histogram(self, tag, data, i):
        if type(data) == torch.Tensor:
            data = data.cpu().detach()
        try:
            wandb.log({tag: wandb.Histogram(data)}, step=i)
        except ValueError as e:
            # you get this eg if there's an inf in `value`, but also in some other cases
            print(f"got ValueError logging `{tag}` histogram to wandb: {e}")

    def add_scalar(self, tag, value, i):
        wandb.log({tag: value}, step=i)

    def add_image(self, tag, value, i):
        wandb.log({tag: [wandb.Image(value, caption="Label")]}, step=i)

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  #noqa
            self.last_time = time.time()
            if i % 10 == 0:
                self.add_scalar("timings/iterations-per-sec", 1/dt, i)
                self.add_scalar("timings/samples-per-sec", batch_size/dt, i)
        except AttributeError:
            self.last_time = time.time()


class ConsoleTracker:
    def __init__(self, name=None, args=None):
        pass

    def add_histogram(self, tag, data, i):
        pass

    def add_scalar(self, tag, value, i):
        print(f"{i}  {tag}: {value}")

    def add_image(self, tag, value, i):
        pass

    def log_iteration_time(self, batch_size, i):
        """Call this once per training iteration."""
        try:
            dt = time.time() - self.last_time  #noqa
            self.last_time = time.time()
            if i % 10 == 0:
                print(f"{i}  iterations-per-sec: {1/dt}")
                print(f"{i}  samples-per-sec: {batch_size/dt}")
        except AttributeError:
            self.last_time = time.time()

class NoTracker:
    def __init__(self, name=None, args=None):
        pass

    def add_histogram(self, tag, data, i):
        pass

    def add_scalar(self, tag, value, i):
        pass

    def add_image(self, tag, value, i):
        pass

    def log_iteration_time(self, batch_size, i):
        pass
