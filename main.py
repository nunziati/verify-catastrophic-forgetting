import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import * 
from nets import *
from classifier import *

def interactive():
    while(True):
        a = input(">>> ")
        if a=="q": break
        exec(a)

def show_example(a):
    plt.imshow(a[0].squeeze())
    plt.show()
    
if __name__ == "__main__":
    device = torch.device("cpu")

    print("Downloading datasets...")
    transform_cifar10 = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Lambda(lambda pic: torch.from_numpy(pic.numpy().transpose((1, 2, 0))))])
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda pic: torch.from_numpy(pic.repeat(3, 1, 1).numpy().transpose((1, 2, 0))))])

    train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    print("Datasets correctly downloaded")

    print("Sorting and merging datasets...")
    train_set = DatasetMerger((train_set_cifar10, train_set_mnist))
    test_set = DatasetMerger((test_set_cifar10, test_set_mnist), set_labels_from=train_set)
    train_set.dataset_wise_sort_by_label()

    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=2)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False, num_workers=2)
    print("Training set correctly sorted!")

    print("Creating models...")
    net = ShallowMLP()
    net.to(torch.float32).to(device)
    
    classifier = Classifier(net)

    classifier.train_class_by_class(train_set_loader, optimizer="adam", lr=0.001, weight_decay=10000000, test_data=test_set_loader)

    classifier.evaluate_class_by_class(test_set_loader, plot=True, always_plot=False)
    print("Models correctly created!")

    print("Starting the training procedure...")
    # for each neural model: train()
    print("Training correctly completed!")

    # plot the result

