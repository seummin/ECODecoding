import os
import re
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_device(cfg):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Device name: cpu')

    return device


def save_checkpoint(cfg, best_epoch, best_model, acc_perplexity):
    if cfg.method == 'post': best_model = best_model.roberta             

    model = best_model.module if hasattr(best_model, "module") else best_model   
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_path, "best_model.bin"))


def load_checkpoint(checkpoint=None, model=None, optimizer=None, scheduler=None):
    print("=> Loading checkpoint")
    print("선언된 모델 : {}".format(len(model.state_dict().keys())))
    print("저장된 모델 : {}".format(len(checkpoint['state_dict'].keys())))

    matched_parameters = len([q for q in list(checkpoint['state_dict'].keys()) if q in list(model.state_dict().keys())])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("{}개 파라미터를 로드했습니다.".format(matched_parameters))
    if not optimizer is None:   optimizer.load_state_dict(checkpoint['optimizer'])
    if not scheduler is None:   scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler


class InverseSqrtScheduler(LambdaLR):
    """ Linear warmup and then follows an inverse square root decay schedule
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Afterward, learning rate follows an inverse square root decay schedule.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class LearningRateSchedule(object):

    def __call__(self, step):
        raise NotImplementedError("Not implemented.")

    def get_config(self):
      raise NotImplementedError("Not implemented.")

    @classmethod
    def from_config(cls, config):
      return cls(**config)



class LinearWarmupRsqrtDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, initial_learning_rate=5e-8,
                 summary=True):
        super(LinearWarmupRsqrtDecay, self).__init__()

        if initial_learning_rate <= 0:
            if warmup_steps > 0:
                initial_learning_rate = learning_rate / warmup_steps
            else:
                initial_learning_rate = 0.0
        elif initial_learning_rate >= learning_rate:
            raise ValueError("The maximum learning rate: %f must be "
                             "higher than the initial learning rate:"
                             " %f" % (learning_rate, initial_learning_rate))

        self._initial_learning_rate = initial_learning_rate
        self._maximum_learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._summary = summary

    def __call__(self, step):
        if step <= self._warmup_steps:
            lr_step = self._maximum_learning_rate - self._initial_learning_rate
            lr_step /= self._warmup_steps
            lr = self._initial_learning_rate + lr_step * step
        else:
            lr = self._maximum_learning_rate

            if self._warmup_steps != 0:
                # approximately hidden_size ** -0.5
                lr = lr * self._warmup_steps ** 0.5

            lr = lr * (step ** -0.5)

        if self._summary:
            summary.scalar("learning_rate", lr, utils.get_global_step())

        return lr

    def get_config(self):
        return {
            "learning_rate": self._maximum_learning_rate,
            "initial_learning_rate": self._initial_learning_rate,
            "warmup_steps": self._warmup_steps
        }