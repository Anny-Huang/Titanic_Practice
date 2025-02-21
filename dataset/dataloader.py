import torch
from dataset.dataset import TitanicDataset
from data.data_processing import proccess_train_data, proccess_test_data  # 確保有 load_test_data

def dataloader(batch_size, file_path, label_path=None, mode="train"):
    """
    加載數據：
    - 訓練和驗證: mode="train"
    - 測試: mode="test" (label_path 可以為 None)
    """
    if mode == "test":
        test_data, test_labels = proccess_test_data(file_path, label_path)  # 加載測試數據
        test_dataset = TitanicDataset(test_data, test_labels)

        testLoader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # 測試時不需要隨機
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
        return testLoader  # 只返回測試 dataloader
    
    # 訓練模式，讀取 train.csv 並切分
    train_data, train_labels, valid_data, valid_labels = proccess_train_data(file_path)
    
    train_dataset = TitanicDataset(train_data, train_labels)
    valid_dataset = TitanicDataset(valid_data, valid_labels)
    
    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )

    validLoader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    return trainLoader, validLoader




