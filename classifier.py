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

    def __init__(self, net="shallow_mlp", device="cpu"):
        """Create a classifier with a specified neural model.

        Args:
            net: a string that should be in net_types, encoding the type of network to use.
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use (can also be torch.device object).
        """
        
        # initialize attributes: device and neural network
        self.device = torch.device(device)
        self.net_type = net_types[net]
        self.net = self.net_type().to(self.device)

        # will contain the history of the evaluation during the class-by-class training procedure
        self.history = None

    def initialize_net(self):
        """Initialization of the network, using its internal initialization method."""

        self.net.initialize()

    def train_class_by_class(self, data, test_data=None, optimizer="adam", lr=0.01, weight_decay=0, plot=True, **kwargs):
        """Train the network using the specified options.
        The training is single-pass, and the evaluation is performed at the end of each class.

        Args:
            data: a dataset or dataloader containg the training data in the form (image, label)
            test_data: a dataset or dataloader (as the previous one) containing the test data
            optimizer: "adam" or "sgd"
            lr: learning rate
            weight_decay: weight multiplying the weight decay regularizaion term
            plot: if True, the history is plotted at the end of the training procedure
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

        # loop on the mini-batches
        for img_mini_batch, label_mini_batch in data:
            # send the mini-batch to the device memory
            img_mini_batch = img_mini_batch.to(self.device)
            label_mini_batch = label_mini_batch.to(self.device)

            # loop over the examples of the current mini-batch
            for img, label in zip(img_mini_batch, label_mini_batch):
                # if we passed to another class, evalute the model on the whole test set and save the results
                if test_data is not None and label.item() != current_label:
                    self.history[current_label] = self.evaluate_class_by_class(test_data)

                # forward step
                # compute the output (actually the logist) of the model on the current example
                _, logits = self.net(img.view((1, *img.shape)))

                # compute the loss function
                loss = loss_function(logits, label.view((1,)))

                # print the loss function, once every 100 epochs
                if i % 100 == 0: print(i, "\t", loss)
                i += 1

                # perform the backward step and the optimization step
                loss.backward()
                opt.step()
                opt.zero_grad()

        # evaluate the model after the examples of the last class have been provided
        self.history[current_label] = self.evaluate_class_by_class(test_data)

        # plot the results, if needed
        if plot: self.plot(**kwargs)     

        # recover the initial train/eval mode
        if not training: self.net.eval()

    def evaluate_class_by_class(self, data):
        """Compute and retrn the accuracy of the classifier on each single class.
        
        Args:
            
        """
        training = self.net.training
        self.net.eval()

        classes = sorted(list(data.dataset.class_map.values()))
        n_classes = len(classes)
        
        true_positive = torch.zeros((n_classes,)).to(self.device)
        total = torch.zeros((n_classes,)).to(self.device)

        for img, label in data:
            img = img.to(self.device)
            label = label.to(self.device)
            output, _ = self.net(img)
            output_label = torch.argmax(output, dim=-1).to(self.device)

            for class_index, c in enumerate(classes):
                true_positive[class_index] += torch.sum(torch.logical_and(label==c, output_label==c))
                total[class_index] += torch.sum(label==c)

        if training: self.net.train()

        return true_positive / total

    def evaluate(self, data):
        training = self.net.training
        self.net.eval()

        correct = 0

        for img, label in data:
            img = img.to(self.device)
            label = label.to(self.device)
            output, _ = self.net(img)
            output_label = torch.argmax(output, dim=-1)
            correct += torch.sum(torch.eq(label, output_label))

        if training: self.net.train()

        return correct / len(data.dataset)

    def plot(self, subplot="class", subdir="", timestamp="", filename1="accuracy_class_by_cass.jpg", filename2="accuracy.jpg"):
        if self.history is None:
            raise Exception("Before plotting, you should train the classifier providing test data.")

        macro_accuracy = self.history.mean(dim=1)

        f = plt.figure(figsize=(15, 12), dpi=300)
        
        if subplot == "class":
            for index, h in enumerate(self.history):
                ax = f.add_subplot(4, 5, index+1)
                ax.bar(range(20), h)
                ax.set_ylim([0, 1])
                ax.set_title("class {}".format(index))
                ax.set_xlabel("time (after epoch #)")
                ax.set_ylabel("accuracy")
        elif subplot == "time":
            for index, h in enumerate(self.history.transpose(0, 1)):
                ax = f.add_subplot(4, 5, index+1)
                ax.bar(range(1, 21), h)
                ax.set_ylim([0, 1])
                ax.set_title("time {}".format(index))
                ax.set_xlabel("class #")
                ax.set_ylabel("accuracy")
        else: raise ValueError("The parameter subplot must be in {'class', 'time'}.")
        
        f.tight_layout(pad=3.0)

        f_macro = plt.figure()
        ax = f_macro.add_subplot(111)
        ax.bar(torch.arange(0, 20), macro_accuracy)
        ax.set_ylim([0, 1])

        if timestamp == "infer": timestamp = str(datetime.now()).replace(" ", "-")[:20]

        f.savefig(subdir + timestamp + "_" + filename1)
        f_macro.savefig(subdir + timestamp + "_" + filename2)

        plt.show()

    def save(self, subdir="", timestamp="", filename="model.pth"):
        classifier_state_dict = {
            "net_type": type(self.net),
            "net": self.net.state_dict(),
            "history": self.history
        }
        
        if timestamp == "infer": timestamp = str(datetime.now()).replace(" ", "-")[:20]

        torch.save(classifier_state_dict, subdir + timestamp + "_" + filename)
    
    def load(self, filename):
        classifier_state_dict = torch.load(filename, map_location=self.device)
        self.net_type = classifier_state_dict["net_type"]
        self.net = net_types[self.net_type]()
        self.net.load_state_dict(classifier_state_dict["net"])
        self.history = classifier_state_dict["history"]

    @classmethod
    def from_file(cls, filename, device="cpu"):
        device = torch.device(device)
        classifier_state_dict = torch.load(filename, map_location=device)
        net = net_types[classifier_state_dict["net_type"]]
        classifier = cls(net, device)
        classifier.net.load_state_dict(classifier_state_dict["net"])
        classifier.history = classifier_state_dict["history"]

        return classifier
