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

if __name__ == "__main__":
    # for computing the execution time
    start_time = time.time()

    # for giving a unique name to the files of the lots and models
    timestamp = str(datetime.now()).replace(" ", "-")[:20]

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
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=2**17, shuffle=False, num_workers=4)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=2**10, shuffle=False, num_workers=4)
    print("Dataloaders are ready.\n")

    print("Creating classifier...")
    # creating a new, empty classifier
    classifier = Classifier("shallow_mlp", device)
    print("Classifier correctly created!")

    print("Training the classifier...")
    # training the classifier
    classifier.train_class_by_class(train_set_loader,
                                    optimizer="sgd",
                                    lr=0.01,
                                    weight_decay=0,
                                    test_data=test_set_loader,
                                    plot=True,
                                    subdir="./figures/",
                                    timestamp=timestamp)
    print("Training correctly completed!")

    print("Evaluating final classifier...")
    # computing the accuracy of the classifier
    accuracy = classifier.evaluate(test_set_loader)
    print("Classifier evaluated...")

    print("Accuracy: {:2.2f}%".format(accuracy*100))

    # pring("Saving the classifier...")
    # classifier.save(subdir="./models/", timestamp=timestamp)
    # pring("Classifier saved.")

    # computing the execution time
    end_time = time.time()
    execution_time = int(end_time - start_time)

    print("PROGRAM IS OVER IN {} SECONDS".format(execution_time))