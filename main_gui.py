######## TO DO (sooner or later): THERE IS NO CONTROL FOR THE TYPES OF THE VALUES COLLECTED FROM THE WIDGETS!!!! ##########
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataset import *
from classifier import *
from datetime import datetime
import PySimpleGUI as sg
from layout import layout
import threading

# to plot the graphs in the window
mpl.use("TkAgg")

def start_program(net_type="shallow_mlp", optimizer="sgd", hidden_units=100, weight_decay=0.0, dropout=0.0, batch_size=64, lr=0.01, device="cpu", savefig=False, save_model=False, L_return=None):
    """Main function called after the selection of the arguments and that runs the main program."""

    # for giving a unique name to the files of the lots and models
    timestamp = str(datetime.now()).replace(" ", "-")[:20]

    print("Program begins.")

     # selecting the device here will change it in the whole program
    device = torch.device(device)

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
    train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Dataloaders are ready.\n")

    print("\nCreating classifier...")
    # creating a new, empty classifier
    classifier = Classifier(net_type, device, dropout=dropout, hidden_units=hidden_units)
    print("Classifier correctly created!")

    print("\nTraining the classifier...")
    # training the classifier
    classifier.train_class_by_class(train_set_loader,
                                    optimizer=optimizer,
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    test_data=test_set_loader,
                                    plot=False,
                                    savefig=False)
    print("Training correctly completed!")

    print("\nEvaluating final classifier...")
    # computing the accuracy of the classifier
    accuracy = classifier.evaluate(test_set_loader)
    print("Classifier evaluated...")

    print("\nAccuracy: {:2.2f}%".format(accuracy*100))

    if save_model:
        print("\nSaving the classifier...")
        classifier.save(subdir="./models/", timestamp=timestamp)
        print("Classifier saved.")
    
    plot = True
    if L_return is not None:
        L_return.append(classifier)
        plot = False

    classifier.plot(savefig=savefig, subdir="./figures/", timestamp=timestamp, plot=plot, figsize=(11, 8.8))

def draw_figure(canvas, figure):
    """Function to draw a figure in a plot inside a pysimplegui window"""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

if __name__ == "__main__":
    # create the window, with the layout of the file "layout.py"
    window = sg.Window(title="Easy launcher", layout=layout, margins=(100, 50)).Finalize()

    # initialization of useful variables for management of thread
    thread = None
    L_return = []

    while True:
        # wait for an event and collect the values of the inputs
        event, values = window.read(timeout=100)

        # when the thread of the computation terminated
        if thread is not None:
            if not thread.is_alive():
                # unpack the returned plots
                classifier = L_return[0]
                
                # return the figures to be plotted
                f, f_macro = classifier.plot(figsize=(11, 8.8))
                
                # draw the plots
                draw_figure(window["-CANVAS1-"].TKCanvas, f)
                draw_figure(window["-CANVAS2-"].TKCanvas, f_macro)

                # update the window
                window.refresh()
                window['Column'].contents_changed()

                # reset the initial conditions
                thread = None
                L_return = []
                
                # enable button
                window["START"].update(disabled=False)

        # presses the START button
        if event == "START":
            # disable button
            window["START"].update(disabled=True)

            # collect the inputs into meaningful variables
            if values[0]:
                net_type = "shallow_mlp"
            elif values[1]:
                net_type = "deep_mlp"
            elif values[2]:
                net_type = "shallow_cnn"
            elif values[3]:
                net_type = "deep_cnn"
            
            if values[4]:
                optimizer = "sgd"
            elif values[5]:
                optimizer = "adam"

            hidden_units = int(values[6])
            weight_decay = float(values[7])
            dropout = float(values[8])
            batch_size = int(values[9])
            lr = float(values[10])

            if values[11]:
                device = "cpu"
            elif values[12]:
                device = "cuda:0"
            elif values[13]:
                device = "cuda:1"
            
            savefig = values[14]
            save_model = values[15]

            # packing the arguments for the main function
            kwargs = {
                "net_type": net_type,
                "optimizer": optimizer,
                "hidden_units": hidden_units,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "batch_size": batch_size,
                "lr": lr,
                "device": device,
                "savefig": savefig,
                "save_model": save_model,
                "L_return": L_return
            }

            # multithread is used for being able to terminate the program closing the window
            thread = threading.Thread(target=start_program, kwargs=kwargs, daemon=True)
            thread.start()

        # if the window is closed
        elif event == sg.WIN_CLOSED:
            break

    window.close()