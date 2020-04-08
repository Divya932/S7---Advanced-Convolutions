import torch
#import torchvision
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def dataloader(trainset, testset, gpu_batch_size, cpu_batch_size, workers, cuda):
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=gpu_batch_size, num_workers=workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=cpu_batch_size)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)


    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return trainloader, testloader