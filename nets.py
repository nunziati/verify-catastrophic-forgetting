import torch


class MyConv2d(torch.nn.Conv2d):
    def __init__(self, in_maps, out_maps, padding="same", **kwargs):
        # this only works for odd kernel_size values or components
        if isinstance(padding, str):
            if padding == "same":
                if isinstance(kwargs["kernel_size"], int):
                    padding = (kwargs["kernel_size"] - 1) // 2
                elif isinstance(kwargs["kernel_size"], tuple):
                    padding = ((x - 1) // 2 for x in kwargs["kernel_size"])
                else: raise TypeError("kernel_size must be int or tuple of int")
            else: raise ValueError("padding str must be 'same'")
        super(MyConv2d, self).__init__(in_maps, out_maps, padding=padding, **kwargs)

class ShallowMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 1000)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(1000, 20)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.sigmoid1(O)
        A = self.linear2(O)
        O = self.softmax(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

class DeepMLP(torch.nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 100)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(100, 100)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(100, 100)
        self.sigmoid3 = torch.nn.Sigmoid()
        self.linear4 = torch.nn.Linear(100, 100)
        self.sigmoid4 = torch.nn.Sigmoid()
        self.linear5 = torch.nn.Linear(100, 100)
        self.sigmoid5 = torch.nn.Sigmoid()
        self.linear6 = torch.nn.Linear(100, 100)
        self.sigmoid6 = torch.nn.Sigmoid()
        self.linear7 = torch.nn.Linear(100, 100)
        self.sigmoid7 = torch.nn.Sigmoid()
        self.linear8 = torch.nn.Linear(100, 100)
        self.sigmoid8 = torch.nn.Sigmoid()
        self.linear9 = torch.nn.Linear(100, 100)
        self.sigmoid9 = torch.nn.Sigmoid()
        self.linear10 = torch.nn.Linear(100, 20)
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.sigmoid1(O)
        O = self.linear2(O)
        O = self.sigmoid2(O)
        O = self.linear3(O)
        O = self.sigmoid3(O)
        O = self.linear4(O)
        O = self.sigmoid4(O)
        O = self.linear5(O)
        O = self.sigmoid5(O)
        O = self.linear6(O)
        O = self.sigmoid6(O)
        O = self.linear7(O)
        O = self.sigmoid7(O)
        O = self.linear8(O)
        O = self.sigmoid8(O)
        O = self.linear9(O)
        O = self.sigmoid9(O)
        A = self.linear10(O)
        O = self.softmax(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

class ShallowCNN(torch.nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        
        self.conv1 = MyConv2d(3, 5, kernel_size=5, padding="same")
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = MyConv2d(5, 10, kernel_size=3, padding="same")
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7*7*10, 20)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, X):
        O = self.conv1(X)
        O = self.relu1(O)
        O = self.maxpool1(O)
        O = self.conv2(O)
        O = self.relu2(O)
        O = self.maxpool2(O)
        O = self.flatten(O)
        A = self.linear(O)
        O = self.softmax(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

class DeepCNN(torch.nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        self.conv1a = torch.nn.Conv2d(3, 5, kernel_size=5, padding="same")
        self.relu1a = torch.nn.ReLU()
        self.conv1b = torch.nn.Conv2d(5, 5, kernel_size=5, padding="same")
        self.relu1b = torch.nn.ReLU()
        self.conv1c = torch.nn.Conv2d(5, 5, kernel_size=5, padding="same")
        self.relu1c = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2a = torch.nn.Conv2d(5, 10, kernel_size=3, padding="same")
        self.relu2a = torch.nn.ReLU()
        self.conv2b = torch.nn.Conv2d(10, 10, kernel_size=3, padding="same")
        self.relu2b = torch.nn.ReLU()
        self.conv2c = torch.nn.Conv2d(10, 10, kernel_size=3, padding="same")
        self.relu2c = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7*7*10, 20)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, X):
        O = self.conv1a(X)
        O = self.relu1a(O)
        O = self.conv1b(O)
        O = self.relu1b(O)
        O = self.conv1c(O)
        O = self.relu1c(O)
        O = self.maxpool1(O)
        O = self.conv2a(O)
        O = self.relu2a(O)
        O = self.conv2b(O)
        O = self.relu2b(O)
        O = self.conv2c(O)
        O = self.relu2c(O)
        O = self.maxpool2(O)
        O = self.flatten(O)
        A = self.linear(O)
        O = self.softmax(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)