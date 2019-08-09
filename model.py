import functools
import logging
import time
from typing import Callable

from catalyst.utils import pack_checkpoint, unpack_checkpoint
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from datasets import get_dataloader
from models import OneHeadNet
from transforms import get_transform_function

logger = logging.getLogger(__name__)


def format_log_message(message, **kwargs):
    metrics_formatted = ['{k}: {v:.3f}'.format(k=k, v=v) for k, v in kwargs.items()]
    metrics_formatted = ' | '.join(metrics_formatted)
    return '\n'.join((message, metrics_formatted))


def timeit(function):
    def warp(*args, **kwargs):
        start_time = time.time()
        results = function(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return results, elapsed_time
    return warp


class Model:
    def __init__(self, metadata):
        self.metadata = metadata

        with open('config.yml') as cfg:
            self.config = yaml.load(cfg, yaml.Loader)

        if torch.cuda.is_available():
            device = self.config.get('device', 'cuda:0')
            benchmark = self.config.get('cubnn_benchmark', True)
            cudnn.benchmark = benchmark
        else:
            device = self.config.get('device', 'cpu')
        self.device = torch.device(device)

        self.config['model_params']['num_classes'] = self.metadata.get_output_size()

        # lr linear scaling rule <-- all auto dl is here
        base_learning_rate = 3e-2
        self.learning_rate = base_learning_rate * (self.config['data_params']['batch_size'] / 256)

        self.get_loader = functools.partial(get_dataloader, **self.config['data_params'])
        for mode in ('train', 'test'):
            transform = self.config['transform'][mode]
            self.config['transform'][mode] = get_transform_function(transform, image_size=metadata.get_matrix_size())

        self.total_epochs = self.config['experiment_params']['n_epochs']
        self.curr_epoch = 0
        self.checkpoint = None
        self.done_training = False

    def train(self, dataset, remaining_time_budget=None):
        if self.done_training:
            return

        self.curr_epoch += 1

        model = OneHeadNet(**self.config['model_params']).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 30, 40], gamma=0.3)
        if self.checkpoint is not None:
            unpack_checkpoint(self.checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)

        data = self.get_loader(
            dataset, transform=self.config['transform']['train'], train=True, epoch_frac=0.5
        )
        train_losses, elapsed_time = self._train_epoch(
            data=data,
            model=model,
            optimizer=optimizer,
            criterion=F.binary_cross_entropy_with_logits,
            scheduler=scheduler
        )
        self.checkpoint = pack_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler)

        msg = '{curr}/{total} * epoch | '.format(curr=self.curr_epoch, total=self.total_epochs)
        msg = format_log_message(msg, train_loss=np.mean(train_losses), elapsed_time=elapsed_time)
        logger.info(msg)

        if self.curr_epoch >= self.total_epochs:
            self.done_training = True

    def test(self, dataset, remaining_time_budget=None):
        model = OneHeadNet(**self.config['model_params']).to(self.device)
        unpack_checkpoint(self.checkpoint, model=model)
        model.eval()

        predictions = []
        data = self.get_loader(dataset, transform=self.config['transform']['test'], train=False)
        with torch.no_grad():
            for i, inputs in enumerate(data):
                sample = self.to_device(inputs['features'])
                y_pred = torch.sigmoid(model(sample)).cpu().numpy()
                predictions.extend(y_pred)

        return np.array(predictions)

    def to_device(self, x):
        if type(x) == dict:
            return {k: self.to_device(v) for k, v in x.items()}
        return x.to(self.device)

    @timeit
    def _train_epoch(
        self,
        data: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Callable,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        clip: float = 1.0
    ):
        model.train()

        losses = []
        for i, inputs in enumerate(data):
            inputs = self.to_device(inputs)
            x, y = inputs['features'], inputs['targets']

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            losses.append(loss.item())
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return losses

    @timeit
    def _valid_epoch(
        self,
        data: DataLoader,
        model: nn.Module,
        criterion: Callable
    ):
        model.eval()

        losses = []
        with torch.no_grad():
            for i, inputs in enumerate(data):
                inputs = self.to_device(inputs)
                x, y = inputs['features'], inputs['targets']

                outputs = model(x)
                loss = criterion(outputs, y)
                losses.append(loss.item())

        return losses
