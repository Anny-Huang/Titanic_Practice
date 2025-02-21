import torch
import torch.nn as nn
# import torch.nn.functional as F

class CeLoss(nn.Module):
    def __init__(self):
        super(CeLoss, self).__init__()
        # self.alpha = 0.25
        # self.gamma = 2
        self.criterion = nn.CrossEntropyLoss() 
        
    # def forward(self, pred, label, weight):
    #     ce = F.cross_entropy(pred, label, weight=weight, reduction='none')
    #     pt = torch.exp(-ce)
    #     focal_loss = (self.alpha * (1-pt)**self.gamma * ce).mean()
    #     return focal_loss
        
    def forward(self, pred, label):
        return self.criterion(pred, label)



# 測試 CeLoss
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CeLoss().to(device)

    # 模擬模型輸出
    pred = torch.randn(16, 2).to(device)  # 16個樣本，2個類別（未生還和生還）
    label = torch.randint(0, 2, (16,)).to(device)  # 隨機生成目標標籤

    # 類別加權（例如生還和未生還的比例）
    weight = torch.tensor([0.7, 1.0]).to(device)  # 可根據類別分佈設置

    # 計算損失
    loss = criterion(pred, label, weight)
    print(f"Focal Loss: {loss.item():.4f}")