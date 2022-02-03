import torch

class ShallowMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 1000)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(1000, 20)
        self.softmax2 = torch.nn.Softmax(dim=0)

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.sigmoid1(O)
        A = self.linear2(O)
        O = self.softmax2(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

class DeepMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 5)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(5, 20)
        self.softmax2 = torch.nn.Softmax()

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.sigmoid1(O)
        A = self.linear2(O)
        O = self.softmax2(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)