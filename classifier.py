import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import nets

class Classifier:
    """Classifier that contains the model and makes predictions."""

    # types of networks that is possible to manage
    net_types = {
        "shallow_mlp": nets.ShallowMLP,
        "deep_mlp": nets.DeepMLP,
        "shallow_cnn": nets.ShallowCNN,
        "deep_cnn": nets.DeepCNN
    }

    def __init__(self, net="shallow_mlp", device="cpu", **kwargs):
        """Create a classifier with a specified neural model.

        Args:
            net: a string that should be in net_types, encoding the type of network to use.
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use (can also be torch.device object).
        """
        
        # initialize attributes: device and neural network
        self.device = torch.device(device)
        self.net_type = Classifier.net_types[net]
        self.net = self.net_type(**kwargs).to(self.device)

        # will contain the history of the evaluation during the class-by-class training procedure
        self.history = None

    def initialize_net(self):
        """Initialization of the network, using its internal initialization method."""

        self.net.initialize()

    def train_class_by_class(self, data, test_data=None, optimizer="adam", lr=0.01, weight_decay=0, plot=False, savefig=False, **kwargs):
        """Train the network using the specified options.
        The training is single-pass, and the evaluation is performed at the end of each class.

        Args:
            data: a dataset or dataloader containg the training data in the form (image, label).
            test_data: a dataset or dataloader (as the previous one) containing the test data.
            optimizer: "adam" or "sgd".
            lr: learning rate.
            weight_decay: weight multiplying the weight decay regularizaion term.
            plot: if True, the history is plotted at the end of the training procedure.
        """

        # save the train/eval mode of the network and change it to training mode
        training = self.net.training
        self.net.train()

        n_classes = data.dataset.num_classes

        # use cross-entropy loss function
        loss_function = torch.nn.CrossEntropyLoss()

        # create the optimizer selected by the caller of the function
        if optimizer == "sgd":
            opt = torch.optim.SGD(self.net.parameters(), lr, weight_decay=weight_decay)
        elif optimizer == "adam":
            opt = torch.optim.Adam(self.net.parameters(), lr, weight_decay=weight_decay)
        else:
            raise ValueError('Optimizer "{}" not defined.'.format(optimizer))

        # select the initial label, assuming it to be 0
        current_label = 0

        # initializing the torch tensor that will contain the history of the evaluation during training
        self.history = torch.empty((n_classes, n_classes), dtype=torch.float32)

        #initilize a counter for the processed training examples
        i = 0

        # loop over the mini-batches
        for img_mini_batch, label_mini_batch in data:
            # send the mini-batch to the device memory
            img_mini_batch = img_mini_batch.to(self.device)
            label_mini_batch = label_mini_batch.to(self.device)

            # loop over the examples of the current mini-batch
            for img, label in zip(img_mini_batch, label_mini_batch):
                # if we passed to another class, evalute the model on the whole test set and save the results
                if test_data is not None and label.item() != current_label:
                    self.history[current_label] = self.evaluate_class_by_class(test_data)
                    current_label = label.item()

                # forward step
                # compute the output (actually the logist) of the model on the current example
                _, logits = self.net(img.view((1, *img.shape)))

                # compute the loss function
                loss = loss_function(logits, label.view((1,)))

                # print the loss function, once every 100 epochs
                if i % 100 == 0: print("Training example = {:<7}\t\t\tLoss = {:<.4f}".format(i, loss.item()))
                i += 1

                # perform the backward step and the optimization step
                loss.backward()
                opt.step()
                opt.zero_grad()

        # evaluate the model after the examples of the last class have been provided
        if test_data is not None:
            self.history[current_label] = self.evaluate_class_by_class(test_data)

        # plot the results, if needed
        if plot or savefig: self.plot(plot=plot, savefig=savefig, **kwargs)     

        # recover the initial train/eval mode
        if not training: self.net.eval()

    def evaluate_class_by_class(self, data):
        """Compute and retrn the accuracy of the classifier on each single class.
        
        Args:
            data: the data set to be used as test set.

        Returns: a torch.Tensor containg the accuracy on each class.
        """
        
        # save the train/eval mode of the network and change it to evaluation mode
        training = self.net.training
        self.net.eval()

        # get the targets and the number of classes
        classes = sorted(list(data.dataset.class_map.values()))
        n_classes = len(classes)
        
        # initializing the counters for the class-by-class accuracy
        true_positive = torch.zeros((n_classes,)).to(self.device)
        total = torch.zeros((n_classes,)).to(self.device)
        
        with torch.no_grad():
            # loop over the mini-batches
            for img, label in data:
                # send the mini-batch to the device memory
                img = img.to(self.device)
                label = label.to(self.device)

                # compute the output of the model on the current mini-batch
                output, _ = self.net(img)

                # decision rule
                output_label = torch.argmax(output, dim=-1).to(self.device)

                # update the counters
                for class_index, c in enumerate(classes):
                    true_positive[class_index] += torch.sum(torch.logical_and(label==c, output_label==c))
                    total[class_index] += torch.sum(label==c)

        # recover the initial train/eval mode
        if training: self.net.train()

        # return the 1D tensor of accuracies of each class
        return true_positive / total

    def evaluate(self, data):
        """Compute and retrn the overall accuracy of the classifier on the whole test set.
        
        Args:
            data: the data set to be used as test set.

        Returns: the accuracy on the whole test set.
        """

        # save the train/eval mode of the network and change it to evaluation mode
        training = self.net.training
        self.net.eval()

        # initializing the counters for the correctly classified examples
        correct = 0

        # loop over the mini-batches
        with torch.no_grad():
            for img, label in data:
                # send the mini-batch to the device memory
                img = img.to(self.device)
                label = label.to(self.device)

                # compute the output of the model on the current mini-batch
                output, _ = self.net(img)

                # decision rule
                output_label = torch.argmax(output, dim=-1)
                
                # update the counters
                correct += torch.sum(torch.eq(label, output_label))

        # recover the initial train/eval mode
        if training: self.net.train()

        # return the accuracy
        return correct / len(data.dataset)

    def plot(self, subplot="class", plot=False, savefig=False, figsize=None, subdir="", timestamp="", filename1="accuracy_class_by_cass.jpg", filename2="accuracy.jpg"):
        """Plotting the results: class-by-class accuracy after each class; overall accuracy after each class.

        Args:
            subplot: if "class", each subplot will represent the accuracy OF a class class; if "time", each subplot is the accuracy AFTER a class.
            salve: if True, the plots will also be saved.
            subdir, timestamp, filename1, filename2: they are useful to decide the path and name of the files to be saved.
        
        Returns: the figures of the plots.
        """

        if self.history is None:
            raise Exception("Before plotting, you should train the classifier providing test data.")

        # computing the overall accuracy over time
        macro_accuracy = self.history.mean(dim=1)
        
        if figsize is None:
            f = plt.figure()
        else:
            f = plt.figure(figsize=figsize)
        

        # looping over the rows/columns of the history of evaluation computed during training, and plotting them on different subplots
        if subplot == "class":
            for index, h in enumerate(self.history):
                ax = f.add_subplot(4, 5, index+1)
                ax.bar(range(20), h)
                ax.set_ylim([0, 1])
                if figsize is not None:
                    ax.set_title("class {}".format(index))
                    ax.set_xlabel("computed after class #")
                    ax.set_ylabel("accuracy")
        elif subplot == "time":
            for index, h in enumerate(self.history.transpose(0, 1)):
                ax = f.add_subplot(4, 5, index+1)
                ax.bar(range(1, 21), h)
                ax.set_ylim([0, 1])
                if figsize is not None:
                    ax.set_title("time {}".format(index))
                    ax.set_xlabel("class #")
                    ax.set_ylabel("accuracy")
        else: raise ValueError("The parameter subplot must be in {'class', 'time'}.")
        
        # add spacing between subplots
        if figsize is not None:
            f.tight_layout(pad=figsize[0]/5)

        # plotting the macro accuracy of the classifier over time
        if figsize is None:
            f_macro = plt.figure()
        else:
            f_macro = plt.figure(figsize=figsize)
        ax = f_macro.add_subplot(111)
        ax.bar(torch.arange(0, 20), macro_accuracy)
        ax.set_ylim([0, 1])

        # saving the plots as figures, using the name provided by the user or using a predefined one
        if savefig:
            if timestamp == "infer": timestamp = str(datetime.now()).replace(" ", "-")[:20]
            
            f.savefig(subdir + timestamp + "_" + filename1)
            f_macro.savefig(subdir + timestamp + "_" + filename2)

        # show the plots
        if plot:
            plt.show()

        return f, f_macro

    def forward(self, X):
        """Compute the output of the network."""
        with torch.no_grad():
            output, logits = self.net(X)
        return output, logits

    def predict(self, X):
        """Compute the output of the classifier: output of the network + decision rule."""
        with torch.no_grad():
            output, _ = self.forward(X)
            label = torch.argmax(output, dim=-1)

        return label

    def save(self, subdir="", timestamp="", filename="model.pth"):
        """Saving the classifier state as a .pth file.
        
        Args:
            subdir, timestamp, filename: they are useful to decide the path and name of the file to be saved.
        """
        # saving the state of the classifier, with all the information to re-build it
        classifier_state_dict = {
            "net_type": self.net_type,
            "net": self.net.state_dict(),
            "history": self.history
        }
        
        # saving the classifier, using the name provided by the user or using a predefined one
        if timestamp == "infer": timestamp = str(datetime.now()).replace(" ", "-")[:20]

        torch.save(classifier_state_dict, subdir + timestamp + "_" + filename)
    
    def load(self, filename):
        """Load a previously saved classifier and use it to fill the state of this classifier.

        Args:
            filename: the .pth file containing the state of the classifier to be loaded.
        """

        # load the file
        classifier_state_dict = torch.load(filename, map_location=self.device)
        
        # check the content of the file
        if self.net_type != classifier_state_dict["net_type"]:
            raise Exception("The classifier in the file is using a different model.")
        
        # if the model is of the correct type, replace the state of the internal classifier
        self.net.load_state_dict(classifier_state_dict["net"])
        self.history = classifier_state_dict["history"]

    @classmethod
    def from_file(cls, filename, device="cpu"):
        """Create a new classifier, loading the state from an external file.
        
        Args:
            filename: the file to be used to load the classifier.
            device: the device on which to build the classifier.

        Returns: an instance of the classifier, build from the file.
        """

        device = torch.device(device)

        # load the file
        classifier_state_dict = torch.load(filename, map_location=device)
        
        # retrieve the type of classifier to build
        net_type = Classifier.net_types[classifier_state_dict["net_type"]]

        # create an instance of the classifier, using the correct net_type and device
        classifier = cls(net_type, device)

        # fill the classifier with the state contained in the source file
        classifier.net.load_state_dict(classifier_state_dict["net"])
        classifier.history = classifier_state_dict["history"]

        # return the classifier
        return classifier
