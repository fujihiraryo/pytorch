import torch
from . import model
from . import dataloader

cnn = model.CNN()
cnn.load_state_dict(torch.load("parameter/cnn.pth"))
testloader = dataloader.testloader
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

class_correct = [0.0 for i in range(10)]
class_total = [0.0 for i in range(10)]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print(
        "Acc of %5s : %2d %%" %
        (classes[i],
         100 *
         class_correct[i] /
         class_total[i]))
