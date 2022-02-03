import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, net):
        self.net = net
        self.history = None

    def initialize_net(self):
        self.net.initialize()

    def train_class_by_class(self, data, test_data=None, optimizer="adam", lr=0.01, weight_decay=0, plot=True, always_plot=False):
        training = self.net.training
        self.net.train()

        n_classes = len(data.dataset.class_map)

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

        for img, label in data:            
            if test_data is not None and label[0] != current_label:
                self.history[current_label] = self.evaluate_class_by_class(test_data, plot)
                current_label = label[0]
            
            _, logits = self.net(img)

            loss = loss_function(logits, label)
            if i % 100 == 0: print(i, "\t", loss)
            i += 1

            loss.backward()
            opt.step()
            opt.zero_grad()

        self.history = [current_label] = self.evaluate_class_by_class(test_data, plot)           

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
            output, _ = self.net(img)
            output_labels = torch.argmax(output, dim=-1)

            total.add_(eye[label].sum(dim=0))
            correct.add_((eye[label]*eye[output_labels]).sum(dim=0))

        if training: self.net.train()

        return correct / total

    def plot(self):
        if self.history is None:
            raise Exception("Before plotting, you should train the classifier.")
        training = self.net.training
        self.net.eval()

        macro_accuracy = self.history.mean(dim=1)

        f = plt.figure()
        for index, h in enumerate(self.history):
            ax = f.add_subplot(4, 5, index+1)
            ax.bar(range(1, 21), h)
            ax.set_ylim([0, 1])
        
        f_macro = plt.figure()
        ax = f_macro.add_subplot(111)
        ax.bar(torch.arange(0, 20), macro_accuracy)
        ax.set_ylim([0, 1])

        plt.show()

        if training: self.net.train()

    def save(self, filename):
        classifier_state_dict = {
            "net": self.net.state_dict(),
            "history": self.net.history
        }

        torch.save(classifier_state_dict, filename)
    
    def load(self, filename):
        classifier_state_dict = torch.load(filename)
        self.net.load_state_dict(classifier_state_dict["net"])
        self.net.history = classifier_state_dict["history"]