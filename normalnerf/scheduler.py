from abc import abstractmethod
from typing import Generic, TypeVar, Callable, Literal, Tuple, List
import math
import numpy as np
import torch

from nerfstudio.models.base_model import Model

T = TypeVar('T')

class Scheduler(Generic[T]):
    @abstractmethod
    def get(self, step: int) -> T:
        pass

    def step_cb(self, model: Model, name: str):
        def func(step: int):
            target = getattr(model, name)
            value = self.get(step)
            if torch.is_tensor(target):
                target[...] = value
            else:
                setattr(model, name, value)
        return func

class ConstantScheduler(Scheduler, Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self, step: int) -> T:
        return self.value

class LambdaScheduler(Scheduler, Generic[T]):
    def __init__(self, lambda_str: str) -> None:
        super().__init__()
        self.lambda_str = "lambda step: " + lambda_str

    def get(self, step: int) -> T:
        return eval(self.lambda_str)(step)


class SequentialScheduler(Scheduler, Generic[T]):
    """
    Example: SequentialScheduler(scheduler1, 1000, scheduler2, 1000, scheduler3)
    """
    def __init__(self, *args) -> None:
        super().__init__()
        self.schedulers: Tuple[Scheduler[T]] = args[::2]    # type: ignore
        self.intervals: Tuple[int] = args[1::2]             # type: ignore

    def get(self, step: int) -> T:
        stop_steps = 0
        for interval, scheduler in zip(self.intervals, self.schedulers):
            if step < stop_steps + interval: 
                return scheduler.get(step - stop_steps)
            stop_steps += interval
        return self.schedulers[-1].get(step - stop_steps)

    
class ExponentialDecayScheduler(Scheduler):
    def __init__(self, init: float, final: float, max_steps: int, warmup_steps: int = 0, pre_warmups: float = 0):
        super().__init__()
        self.init = init
        self.final = final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.pre_warmups = pre_warmups

    def get(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.pre_warmups + (self.init - self.pre_warmups) * np.sin(0.5 * np.pi * np.clip(step / self.warmup_steps, 0, 1))
        else:
            t = np.clip((step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 0, 1)
            return np.exp(np.log(self.init) * (1 - t) + np.log(self.final) * t)
    
class AnnealingWarmRestarts(Scheduler):
    def __init__(self, max: float, min: float, T_0: int, T_mult: int, mode: Literal["cosine", "exp"]):
        super().__init__()
        self.max = max
        self.min = min
        self.T_0 = T_0
        self.T_mult = T_mult
        self.mode = mode

    def get(self, step: int) -> float:
        if step >= self.T_0:
            if self.T_mult == 1:
                t = (step % self.T_0) / self.T_0
            else:
                n = int(math.log((step / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                t = step - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                t_max = self.T_0 * self.T_mult ** (n)
                t = t / t_max
        else:
            t = step / self.T_0
        if self.mode == "cosine":
            return self.min + (self.max - self.min) * (1 + np.cos(np.pi * t)) / 2
        else:
            return np.exp(np.log(self.max) * (1 - t) + np.log(self.min) * t)
        

class OneCycleScheduler(Scheduler):
    def __init__(self, init: float, final: float, max_steps: int, warmup_steps: int = 0, pre_warmups: float = 0):
        super().__init__()
        self.init = init
        self.final = final
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.pre_warmups = pre_warmups

    def get(self, step: int) -> float:
        if step < self.warmup_steps:
            t = np.clip(step / self.warmup_steps, 0, 1)
            return self.pre_warmups + (self.init - self.pre_warmups) * np.sin(0.5 * np.pi * t)
        else:
            t = np.clip((step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 0, 1)
            return self.final + (self.init - self.final) * (np.cos(np.pi * t) + 1) / 2