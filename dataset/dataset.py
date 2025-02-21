import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data  
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = torch.tensor(self.data.iloc[index].values, dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long) if self.labels is not None else torch.tensor(-1)
        return features, label










