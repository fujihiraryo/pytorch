import torch
import torch.nn as nn
import torch.optim as optim

import model
import dataloader
import settings

import matplotlib.pyplot as plt
from datetime import datetime as dt

cnn = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(),
                      lr=settings.lr,
                      momentum=settings.momentum)
trainloader = dataloader.trainloader
testloader = dataloader.testloader

train_loss = []
test_loss = []

for epoch in range(settings.epochs):
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    for i, data in enumerate(trainloader):
        inputs, labels = data
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())

K = len(train_loss)
plt.plot(range(K), train_loss, label='train')
plt.plot(range(K), test_loss, label='test')
plt.legend()
now = dt.now().strftime("%Y%m%d%H%M%S")
plt.savefig('result/'+now+'.png')
