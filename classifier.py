import torch

class Classifier:
    def __init__(self, net):
        self.net = net
        
    def initialize_net(self):
        net.initialize()

    def train_class_by_class(net, data, optimizer="sgd", lr=0.01, plot=False):
        training = net.training
        net.train()

        loss_function = torch.nn.CategoricalCrossEntropy()

        if optimizer == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr)
        elif optimizer == "adam":
            opt = torch.optim.Adam(net.parameters(), lr)
        else:
            raise ValueError('Optimizer "{}" not defined.'.format(optimizer))

        _, current_label = data[0]

        for img, label in data:
            if label != current_label:
                show_partial_results()
            
            output = net(img)

            loss = loss_function(output, label)

    def show_partial_results(net, data, current_label)