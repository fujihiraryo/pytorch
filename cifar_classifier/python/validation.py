# standard packages
import time
from datetime import datetime as dt

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets

# my modules
import model
import dataloader
import settings
import utils

K = settings.folds
cnn = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=settings.lr, momentum=settings.momentum)
trainset = dataloader.trainset

start = time.time()
for k in range(K):
    train_loss, val_loss = [], []
    trainloader, valloader = utils.train_val_split(trainset, K)
    print("data loaded")
    for epoch in range(settings.epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        for i, data in enumerate(valloader):
            inputs, labels = data
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            val_loss.append(loss.item())
        print(train_loss[-1], val_loss[-1])
    mean_train_loss = sum(train_loss) / len(train_loss)
    mean_val_loss = sum(val_loss) / len(val_loss)
    now = time.time() - start
    text = "fold:{:3d}  train:{:3.3f}  val:{:3.3f}  time:{:5.3f}".format(
        k + 1, mean_train_loss, mean_val_loss, now
    )
    print(text)
