import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import *
from nets import *
from classifier import *
import time

def interactive():
    while(True):
        a = input(">>> ")
        if a=="q": break
        exec(a)
    
if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda:0")

    print("Downloading datasets...")
    transform_cifar10 = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda pic: pic.repeat(3, 1, 1))])

    train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    print("Datasets correctly downloaded")

    print("Sorting and merging datasets...")
    train_set = DatasetMerger((train_set_cifar10, train_set_mnist))
    test_set = DatasetMerger((test_set_cifar10, test_set_mnist), set_labels_from=train_set)
    train_set.dataset_wise_sort_by_label()

    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=2**17, shuffle=False, num_workers=4)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=2**17, shuffle=False, num_workers=4)
    print("Training set correctly sorted!")

    print("Creating classifier...")
    net = ShallowMLP()
    net.to(torch.float32)
    
    classifier = Classifier(net, device)
    print("Classifier correctly created!")

    print("Starting the training procedure...")
    classifier.train_class_by_class(train_set_loader, optimizer="sgd", lr=0.00003, weight_decay=0.3, test_data=test_set_loader)
    print("Training correctly completed!")

    classifier.save("test.pth")
    
    print("Evaluating classifier...")
    accuracy = classifier.evaluate(test_set_loader)
    print("Classifier evaluated...")

    print("Accuracy: {:2.2f}%".format(accuracy*100))

    classifier.plot()

    end_time = time.time()

    print("PROGRAM IS OVER IN {} SECONDS".format(int(end_time-start_time)))