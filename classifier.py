import torch

class Classifier:
    def __init__(self, net):
        self.net = net

    def initialize_net(self):
        net.initialize()

    def train_class_by_class(self, data, test_set=None, optimizer="sgd", lr=0.01, plot=True, always_plot=False):
        training = self.net.training
        self.net.train()

        loss_function = torch.nn.CrossEntropyLoss()

        if optimizer == "sgd":
            opt = torch.optim.SGD(self.net.parameters(), lr)
        elif optimizer == "adam":
            opt = torch.optim.Adam(self.net.parameters(), lr)
        else:
            raise ValueError('Optimizer "{}" not defined.'.format(optimizer))

        current_label = 0 # better way to have this?
        
        for i, (img, label) in enumerate(data):
            print(i)
            if test_set is not None and (label != current_label or always_plot):
                self.evaluate_class_by_class(data, plot, always_plot)

            output = self.net(img)

            loss = loss_function(output, label)
            loss.backward()
            opt.zero_grad()
            opt.step()

        if not training: self.net.eval()

    def evaluate_class_by_class(self, data, plot=True, always_plot=False):
        training = self.net.training
        self.net.eval()




        if training: self.net.train()