# Verify catastrophic forgetting
Simple experiment for verifying the phenomenon of catastrophic forgetting, for a bunch of different neural network architectures, on a dataset realized merging CIFAR-10 and MNIST datasets.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/nunziati/verify-catastrophic-forgetting.git
```
2. Navigate to the directory of the repository
3. Create a virtual environment (optional, but suggested):
4. Activate the virtual environment
5. Install the required python modules (torch, torchvision, matplotlib, pysimplegui)
\
...or...
\
\
Simply copy-paste the following commands in a terminal:
```bash
git clone https://github.com/nunziati/verify-catastrophic-forgetting.git
cd verify-catastrophic-forgetting
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run the code
you can run a command-line-based interface or a graphical interface.

Note: if you want to save the trained model or some figures, just be sure that the target directory exists,
because your OS may forbid python to create new directories.
The default subdirectories are "./models/" and "./figures/".
```bash
mkdir models
mkdir figures
```
### Command line version
For running the code from the command line, you have to run the command:
```bash
python main.py
```
Some options can be specified for customizing the run:
```
--model             the name of the model to be used for the classifier
    (optional)
    choices = "shallow_mlp", "deep_mlp", "shallow_cnn", "deep_cnn"
    default = "shallow_mlp"

--optimizer         the optimizer to use in the model
    (optional)
    choices = "sgd", "adam"
    default = "sgd"

--hidden            number of hidden units in the hidden layers (works only with MLP models)
    (optional)
    type = int
    default = 100

--weight_decay      regularization parameter (default: 0.0)
    (optional)
    type = float
    default = 0.0

--dropout           dropout probability (default 0.0) (works only with MLP models)
    (optional)
    type = float
    default=0.0

--batch_size        mini-batch size (default: 64)
    (optional)
    type = int
    default = 64

--lr                learning rate (default: 0.001)
    (optional)
    type = float
    default = 0.001

--device            device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)
    (optional)
    type = str
    default = "cpu"

--plot              to plot the results of the training procedure
    (optional)

--save              to save the final model obtained
    (optional)

--savefig           to save the plots of the results of the training procedure
    (optional)
```
### GUI version
For using the graphical interface, just open the file "main_gui.py".
The interface allows the customization of the main parameters.
Press START to begin the trainig procedure.

### Demo
The repository contains a demo that you can run without the effort of selecting the configuration.
The parameters have been selected for getting a fast demonstration that can be run without GPU.
```bash
python demo.py
```

## Code structure
The code is organized in different modules, the ones implementing the logic of the project are *nets.py*, *dataset.py*, *classifier.py*.
### nets.py
Contains the definition of the classes representing the neural models.

4 models are defined there: ShallowMLP, DeepMLP, ShallowCNN, DeepCNN.

An additional model (MyConv2d) is a subclass of the base pytorch class implementing a 2d convolution.
Its purpose is to make the otpion padding="same" compatible with older versions of pytorch.

### dataset.py
It contains the class that is responsible to realize a dataset that merges other datasets.

The merging is implemented by assigning unique identifiers to each esample of each dataset and keeping an iterable of those identifiers.

It is responsible for sorting the dataset by class.

### classifier.py
It implements the classifier class, that contains a neural network model.
It can:

- use a training set for training
- while training, use a test set for evaluating the network in each phase of the training (after each class)
- evaluate the performances of a trained model
- perform prediction
- saving and loading the model
- building a model, directly from a file

### Other modules
Other modules are present in the repository: they are for presentation of the results and experimental activity.