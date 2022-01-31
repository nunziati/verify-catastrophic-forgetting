import torch

class ShallowMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 100)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(100, 100)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(100, 20)
        self.softmax3 = torch.nn.Softmax(dim=1)

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.sigmoid1(O)
        O = self.linear2(O)
        O = self.sigmoid2(O)
        O = self.linear3(O)
        O = self.softmax3(O)

        return O
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)    