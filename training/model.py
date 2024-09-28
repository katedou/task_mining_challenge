import torch.nn as nn


class ShallowNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(ShallowNet, self).__init__()
        layers = []
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_dim = h
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
