# pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# my modules
import model
import dataloader
import settings

# standard packages
import matplotlib.pyplot as plt
from datetime import datetime as dt
import time

cnn = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=settings.lr, momentum=settings.momentum)
trainloader = dataloader.trainloader
testloader = dataloader.testloader

train_loss = []
test_loss = []

start = time.time()
for epoch in range(settings.epochs):
    train_loss_epoch = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    train_loss.append(train_loss_epoch / (i + 1))
    test_loss_epoch = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        test_loss_epoch += loss.item()
    test_loss.append(test_loss_epoch / (i + 1))

    text = "epoch:{:3d}  train:{:3.3f}  test:{:3.3f}  time:{:5.3f}".format(
        epoch + 1, train_loss[-1], test_loss[-1], time.time() - start
    )
    print(text)

K = len(train_loss)
plt.plot(range(K), train_loss, label="train")
plt.plot(range(K), test_loss, label="test")
plt.legend()
now = dt.now().strftime("%Y%m%d%H%M%S")
plt.savefig("result/{}.png".format(now))
