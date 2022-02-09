import torch
import torchvision
import torchvision.transforms as transforms
import sys

sys.path.append('..')
from dataset import *
from nets import *
from classifier import *

if __name__ == "__main__":
    output_filename = sys.argv[1]
    start_experiment = int(sys.argv[2])
    end_experiment = int(sys.argv[3])

    # selecting the device here will change it in the whole program
    device = torch.device("cuda:1")

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

    mlp_list = ["shallow_mlp", "deep_mlp"]
    mlp_hidden_units = [10, 100, 1000, 10000]
    optimizer = ["sgd", "adam"]
    weight_decay_list = [0, 3e-2, 3e-1, 1, 3, 10]
    dropout_list = [0, 0.5, 0.8]
    lr_list = [1e-7, 1e-5, 1e-3, 1e-1]

    # creating the combinatios of parameters for grid serach
    grid_search_params = [(x, y, z, u, v, w)
        for x in mlp_list
        for y in mlp_hidden_units
        for z in optimizer
        for u in weight_decay_list
        for v in dropout_list
        for w in lr_list
    ]

    for index, (x, y, z, u, v, w) in enumerate(grid_search_params):
        if index < start_experiment: continue
        if index >= end_experiment: break

        print("Experiment number:", index)
        print("Parameters:")
        print("\tNetwork type:", x)
        print("\tNumber of hidden units:", y)
        print("\tOptimizer:", z)
        print("\tRegularization parameter:", u)
        print("\tDropout probability:", v)
        print("\tLearning rate:", w)

        classifier = Classifier(x, device, hidden_units=y, dropout=v)

        classifier.train_class_by_class(train_set_loader,
            optimizer=z,
            lr=w,
            weight_decay=u,
            plot=False)

        accuracy = classifier.evaluate(test_set_loader)

        with open(output_filename, "a+") as f:
            f.write(",".join([str(value) for value in (x, y, z, u, v, w, accuracy)]) + "\n")

        del classifier
        del accuracy
