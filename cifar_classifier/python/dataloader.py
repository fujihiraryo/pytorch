import torch
import torchvision
import torchvision.transforms as transforms
from . import settings

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(settings.mean, settings.sigma)]
)
trainset = torchvision.datasets.CIFAR10(
    root="./data/", train=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root="./data/", train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=settings.num_workers,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=settings.batch_size,
    shuffle=True,
    num_workers=settings.num_workers,
)
