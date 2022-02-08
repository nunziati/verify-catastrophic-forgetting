import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import *
from nets import *
from classifier import *
import time
from datetime import datetime
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    start_experiment = int(sys.argv[2])
    end_experiment = int(sys.argv[3])

    # selecting the device here will change it in the whole program
    device = torch.device("cuda:0")

    print("Downloading and preparing single datasets...")
    # for the cifar10 dataset, the pictures are resized to the size of the pictures in the mnist dataset
    transform_cifar10 = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])

    # for the mnist dataset, the pictures are turned from grayscale to RGB, by adding 2 identical channels
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda pic: pic.repeat(3, 1, 1))])

    # creating the training and test set of each dataset
    train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    print("Single datasets are ready.\n")

    print("Creating merged datasets...")
    # merging the datasets
    train_set = DatasetMerger((train_set_cifar10, train_set_mnist))
    test_set = DatasetMerger((test_set_cifar10, test_set_mnist), set_labels_from=train_set)
    print("Merged datasets are ready.\n")

    print("Sorting merged test set...")
    # sorting the merged training set
    train_set.dataset_wise_sort_by_label()
    print("Merged test set correctly sorted.\n")

    print("Creating dataloaders...")
    # creating dataloaders for the training and test set
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=2**12, shuffle=False, num_workers=4)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=2**10, shuffle=False, num_workers=4)
    print("Dataloaders are ready.\n")

    cnn_list = ["shallow_cnn", "deep_cnn"]
    optimizer = ["sgd", "adam"]
    weight_decay_list = [0, 3e-2, 1e-1, 3e-1, 1, 3, 10]
    lr_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # creating the combinatios of parameters for grid serach
    grid_search_params = [(x, z, u, w)
        for x in cnn_list
        for z in optimizer
        for u in weight_decay_list
        for w in lr_list
    ]

    for index, (x, z, u, w) in enumerate(grid_search_params):
        if index < start_experiment: continue
        if index >= end_experiment: break

        print("Experiment number:", index)
        print("Parameters:")
        print("\tNetwork type:", x)
        print("\tOptimizer:", z)
        print("\tRegularization parameter:", u)
        print("\tLearning rate:", w)

        classifier = Classifier(x, device)

        classifier.train_class_by_class(train_set_loader,
            optimizer=z,
            lr=w,
            weight_decay=u,
            plot=False)

        accuracy = classifier.evaluate(test_set_loader)

        with open(filename, "a+") as f:
            f.write(",".join([str(value) for value in (x, z, u, w, accuracy)]) + "\n")

        del classifier
        del accuracy
