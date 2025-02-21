import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_cls):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(   
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Batch Normalization after the first layer
            nn.ReLU(),
            nn.Linear(64, num_cls), 

            # nn.Linear(input_dim, 64),          
            # nn.ReLU(),  
            # nn.LayerNorm(64),  # 使用 LayerNorm 而不是 BatchNorm

            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.LayerNorm(32),  # 使用 LayerNorm
            # nn.Dropout(0.3),  # Dropout 獨立存在

            # nn.Linear(32, num_cls)  # 輸出層無 LayerNorm / Dropout
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


