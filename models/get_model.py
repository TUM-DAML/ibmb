from typing import Optional, Union

import torch

from .GAT import GAT
from .GCN import GCN
from .SAGE import SAGEModel


def get_model(graphmodel: str,
              num_node_features: int,
              num_classes: int,
              hidden_channels: int,
              num_layers: int,
              heads: Optional[int],
              device: Union[torch.device, str]):
    if graphmodel == 'gcn':
        model = GCN(num_node_features=num_node_features,
                    num_classes=num_classes,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers)

    elif graphmodel == 'gat':
        model = GAT(in_channels=num_node_features,
                    hidden_channels=hidden_channels,
                    out_channels=num_classes,
                    num_layers=num_layers,
                    heads=heads)
    elif graphmodel == 'sage':
        model = SAGEModel(num_node_features=num_node_features,
                          num_classes=num_classes,
                          hidden_channels=hidden_channels,
                          num_layers=num_layers)
    else:
        raise ValueError

    return model.to(device)
