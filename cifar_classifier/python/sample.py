import numpy as np
import matplotlib.pyplot as plt
import torchvision
import dataloader

trainloader = dataloader.trainloader
testloader = dataloader.testloader
images, labels = iter(trainloader).next()
images, labels = images[:25], labels[:25]
grid_image = torchvision.utils.make_grid(images, nrow=5, padding=1)
grid_image = grid_image.numpy()
grid_image = grid_image / 2 + 0.5
grid_image = np.transpose(grid_image, (1, 2, 0))
plt.imsave(".././image/sample.png", grid_image)
