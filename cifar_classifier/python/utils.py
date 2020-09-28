import random
import torch
from . import settings


def train_val_split(dataset, K):
    N = len(dataset)
    train_indices = random.sample(range(N), N // K)
    val_indices = list(set(range(N)) - set(train_indices))
    trainset = torch.utils.data.dataset.Subset(dataset, train_indices)
    valset = torch.utils.data.dataset.Subset(dataset, val_indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=settings.batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=settings.batch_size, shuffle=True
    )
    return trainloader, valloader
