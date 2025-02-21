import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_cls):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),  
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 🔹Batch Normalization 放在前面，穩定輸入數據
            
            nn.Linear(64, 32),
            nn.ReLU(),            
            nn.Dropout(0.2),  # 🔹Dropout 放在後面，防止過擬合
            
            nn.Linear(32, num_cls)  # 🔹輸出層
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


