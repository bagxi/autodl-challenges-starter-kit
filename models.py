import collections

from pytorchcv import model_provider
import torch
from torch import nn
from torch.nn import functional as F


class OneHeadNet(nn.Module):
    def __init__(self, encoder_params: dict, num_classes: int, dropout: float = 0.2):
        super().__init__()

        model = model_provider.get_model(**encoder_params)
        self.backbone = model.features
        self.backbone.final_pool = nn.AdaptiveAvgPool2d((1, 1))

        _, n_features, *_ = self.backbone(torch.rand(2, 3, 32, 32)).size()  # backbone last layer `out_features`
        self.linear = nn.Sequential(collections.OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc', nn.Linear(n_features, num_classes)),
        ]))

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear(x)
        return x
