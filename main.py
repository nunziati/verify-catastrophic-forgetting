import torch
import torchvision
import torchvision.transforms as transforms
from dataset import *
from classifier import *
import time
from datetime import datetime
import argparse

def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default="shallow_mlp", choices=["shallow_mlp", "deep_mlp", "shallow_cnn", "deep_cnn"], help="the name of the model to be used for the classifier")
    parser.add_argument('--optimizer', default="sgd", choices=["sgd", "adam"], help="the optimizer to use in the model")
    parser.add_argument("--hidden", type=int, default=100, help="number of hidden units in the hidden layers (works only with MLP models)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="regularization parameter (default: 0.0)")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout probability (default 0.0) (works only with MLP models)")
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
    parser.add_argument("--device", default="cpu", type=str, help="device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)")
    parser.add_argument('--plot', dest='plot', action='store_true', default=False)
    parser.add_argument('--save', dest='save', action='store_true', default=False)
    parser.add_argument('--savefig', dest='savefig', action='store_true', default=False)

    parsed_arguments = parser.parse_args()

    # do i need this?
    # parsed_arguments.splits = splits

    return parsed_arguments

if __name__ == "__main__":
    # for computing the execution time
    start_time = time.time()

    # for giving a unique name to the files of the lots and models
    timestamp = str(datetime.now()).replace(" ", "-")[:20]

    print("Program begin.")

    # parsing and printing the arguments passed from command line
    args = parse_command_line_arguments().__dict__
    for k, v in args.items():
        print(k + '=' + str(v))

    # selecting the device here will change it in the whole program
    device = torch.device(args["device"])

    _ = input("\nIf the datasets are not present, they will be downloaded.\nPress enter to proceed...")

    print("\nDownloading and preparing single datasets...")
    # for the cifar10 dataset, the pictures are resized to the size of the pictures in the mnist dataset
    transform_cifar10 = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])

    # for the mnist dataset, the pictures are turned from grayscale to RGB, by adding 2 identical channels
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda pic: pic.repeat(3, 1, 1))])

    # creating the training and test set of each dataset
    train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
    test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
    train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
    test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
    print("Single datasets are ready.")

    print("\nCreating merged datasets...")
    # merging the datasets
    train_set = DatasetMerger((train_set_cifar10, train_set_mnist))
    test_set = DatasetMerger((test_set_cifar10, test_set_mnist), set_labels_from=train_set)
    print("Merged datasets are ready.")

    print("\nSorting merged test set...")
    # sorting the merged training set
    train_set.dataset_wise_sort_by_label()
    print("Merged test set correctly sorted.\n")

    print("\nCreating dataloaders...")
    # creating dataloaders for the training and test set
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], shuffle=False, num_workers=4)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=4)
    print("Dataloaders are ready.\n")

    print("\nCreating classifier...")
    # creating a new, empty classifier
    classifier = Classifier(args["model"], device, dropout=args["dropout"], hidden_units=args["hidden"])
    print("Classifier correctly created!")

    _ = input("\nThe classifier is ready to be trained.\nPress enter to proceed...")

    print("\nTraining the classifier...")
    # training the classifier
    classifier.train_class_by_class(train_set_loader,
                                    optimizer=args["optimizer"],
                                    lr=args["lr"],
                                    weight_decay=args["weight_decay"],
                                    test_data=test_set_loader,
                                    plot=args["plot"],
                                    savefig=args["savefig"],
                                    subdir="./figures/",
                                    timestamp=timestamp)
    print("Training correctly completed!")

    print("\nEvaluating final classifier...")
    # computing the accuracy of the classifier
    accuracy = classifier.evaluate(test_set_loader)
    print("Classifier evaluated...")

    print("\nAccuracy: {:2.2f}%".format(accuracy*100))

    if args["save"]:
        print("\nSaving the classifier...")
        classifier.save(subdir="./models/", timestamp=timestamp)
        print("Classifier saved.")

    # computing the execution time
    end_time = time.time()
    execution_time = int(end_time - start_time)

    print("\nProgram completed in {} seconds.".format(execution_time))
