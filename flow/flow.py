import torch
import numpy as np
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calculate_accuracy(logits, target):
    logits = torch.softmax(logits, dim=-1).argmax(dim=-1)
    correct = logits.eq(target.data).cpu().sum()
    accuracy = correct / float(target.size(0))
    return accuracy

def class_weight(target, num_cls=2):
    target=target.numpy(force=True)
    unique, counts = np.unique(np.array(target), return_counts=True)
    num_class = dict(zip(unique, counts))
    alpha = np.ones((num_cls,))
    
    for c in num_class.keys():
        w = target.shape[0] / (num_class[c] * len(num_class))
        alpha[ c ] = w

    alpha = alpha.astype(np.float32)
    return torch.from_numpy(alpha)

def train(now_ep, model, optimizer, dataloader, criterion, DEVICE):
    losses = []
    accuracies = []
    model.train()
    with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
        for features, target in loader:
            loader.set_description(f"train {now_ep}")

            features, target = features.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            pred = model(features)
            # alpha = class_weight(target)
            # loss = criterion(pred, target, alpha.to(DEVICE))

            loss = criterion(pred, target)
            
            lr = get_lr(optimizer)
            accuracy = calculate_accuracy(pred, target)
            
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            
            loss.backward()
            optimizer.step()

            loader.set_postfix(loss=np.mean(losses), accuracy=np.mean(accuracies), lr=lr)
    return np.mean(losses), np.mean(accuracies)

def evaluate(mode, model, dataloader, criterion, DEVICE):
    model.eval()
    losses = [] if mode.lower() == "validation" else None  
    accuracies = []

    with torch.no_grad():
        with tqdm(dataloader, ascii=' =', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as loader:
            for features, target in loader:
                loader.set_description(f"{mode}")
                features, target = features.to(DEVICE), target.to(DEVICE)

                pred = model(features)
                accuracy = calculate_accuracy(pred, target)
                accuracies.append(accuracy.item())

                if losses is not None: 
                    loss = criterion(pred, target)
                    losses.append(loss.item())
                    loader.set_postfix(loss=np.mean(losses), accuracy=np.mean(accuracies))
                else:
                    loader.set_postfix(accuracy=np.mean(accuracies))

    avg_loss = np.mean(losses) if losses is not None else None
    avg_accuracy = np.mean(accuracies)

    return avg_loss, avg_accuracy


