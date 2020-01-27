import torch
import torch.nn as nn
import torch.optim as optim

import model
import dataloader

cnn = model.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
trainloader = dataloader.trainloader

for epoch in range(settings.epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
torch.save(cnn.state_dict(), './parameter/cnn.pth')
