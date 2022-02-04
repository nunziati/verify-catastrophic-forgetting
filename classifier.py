import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

class Classifier:
    def __init__(self, net, device="cpu"):
        self.net = net

        self.device = torch.device(device)
        self.history = None

        self.net.to(self.device)

    def initialize_net(self):
        self.net.initialize()

    def train_class_by_class(self, data, test_data=None, optimizer="adam", lr=0.01, weight_decay=0, plot=True):
        training = self.net.training
        self.net.train()

        n_classes = data.dataset.num_classes

        loss_function = torch.nn.CrossEntropyLoss()

        if optimizer == "sgd":
            opt = torch.optim.SGD(self.net.parameters(), lr, weight_decay=weight_decay)
        elif optimizer == "adam":
            opt = torch.optim.Adam(self.net.parameters(), lr, weight_decay=weight_decay)
        else:
            raise ValueError('Optimizer "{}" not defined.'.format(optimizer))

        current_label = 0 # better way to have this?
        self.history = torch.empty((n_classes, n_classes), dtype=torch.float32)

        i = 0

        for img_mini_batch, label_mini_batch in data:
            img_mini_batch = img_mini_batch.to(self.device)
            label_mini_batch = label_mini_batch.to(self.device)

            for img, label in zip(img_mini_batch, label_mini_batch):
                if test_data is not None and label.item() != current_label:
                    self.history[current_label] = self.evaluate_class_by_class(test_data, plot)
                    current_label = label.item()
                _, logits = self.net(img.view((1, *img.shape)))
                loss = loss_function(logits, label.view((1,)))
                if i % 100 == 0: print(i, "\t", loss)
                i += 1

                loss.backward()
                opt.step()
                opt.zero_grad()

        self.history[current_label] = self.evaluate_class_by_class(test_data, plot)           

        if not training: self.net.eval()

    def evaluate_class_by_class(self, data, plot=True, always_plot=False):
        training = self.net.training
        self.net.eval()

        classes = sorted(list(data.dataset.class_map.values()))
        n_classes = len(classes)
        
        eye = torch.eye(len(classes))
        total = torch.zeros((n_classes,))
        correct = torch.zeros((n_classes,))

        for img, label in data:
            img = img.to(self.device)
            label = label.to(self.device)
            output, _ = self.net(img)
            output_label = torch.argmax(output, dim=-1)

            total.add_(eye[label].sum(dim=0))
            correct.add_((eye[label]*eye[output_label]).sum(dim=0))

        if training: self.net.train()

        return correct / total

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

    def plot(self, subplot="class"):
        if self.history is None:
            raise Exception("Before plotting, you should train the classifier providing test data.")

        macro_accuracy = self.history.mean(dim=1)

        f = plt.figure(figsize=(15, 12), dpi=300)
        
        if subplot == "class":
            for index, h in enumerate(self.history):
                ax = f.add_subplot(4, 5, index+1)
                ax.bar(range(1, 21), h)
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

        timestamp = str(datetime.now()).replace(" ", "-")[:20]
        f.savefig("./figures/" + timestamp + "accuracy_class_by_cass.jpg")
        f_macro.savefig("./figures/" + timestamp + "_accuracy.jpg")
        
        plt.show()

    def save(self, filename):
        classifier_state_dict = {
            "net_type": type(self.net),
            "net": self.net.state_dict(),
            "history": self.history
        }

        torch.save(classifier_state_dict, filename)
    
    def load(self, filename):
        classifier_state_dict = torch.load(filename, map_location=self.device)
        self.net = classifier_state_dict["net_type"]()
        self.net.load_state_dict(classifier_state_dict["net"])
        self.history = classifier_state_dict["history"]

    @staticmethod
    def load_from_file(filename, device="cpu"):
        device = torch.device(device)
        classifier_state_dict = torch.load(filename, map_location=device)
        net = classifier_state_dict["net_type"]()
        classifier = Classifier(net, device)
        classifier.net.load_state_dict(classifier_state_dict["net"])
        classifier.history = classifier_state_dict["history"]

        return classifier
