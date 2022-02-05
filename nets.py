import torch

class MyConv2d(torch.nn.Conv2d):
    """Custom alias for torch.nn.Conv2d, built because paddint="same" is not a valid option in some versions of pytorch.
        There is no need of a forward method, because torch.nn.Conv2d already defines it."""

    def __init__(self, in_maps, out_maps, padding="same", **kwargs):
        """Constructor of the class is the same as the constructor of the parent class.
        
        Args:
            the same as for torch.nn.Conv2d
            padding: if "same", not sure if it works for even values of kernel_size
        """

        if isinstance(padding, str):
            if padding == "same":
                # computing the padding (or padding tuple) in a way that the input and output diensions are the same
                if isinstance(kwargs["kernel_size"], int):
                    padding = (kwargs["kernel_size"] - 1) // 2
                elif isinstance(kwargs["kernel_size"], tuple):
                    padding = ((x - 1) // 2 for x in kwargs["kernel_size"])
                else:
                    raise TypeError("kernel_size must be int or tuple of int")
            else:
                raise ValueError("padding str must be 'same'")

        # use the computed padding and the other parameters to initialize the instance of torch.nn.Conv2d
        super(MyConv2d, self).__init__(in_maps, out_maps, padding=padding, **kwargs)

class ShallowMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, 1000)
        self.activation1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.8)
        self.linear2 = torch.nn.Linear(1000, 20)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.activation1(O)
        O = self.dropout1(O)
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
        self.activation1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(100, 100)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(100, 100)
        self.activation3 = torch.nn.Sigmoid()
        self.linear4 = torch.nn.Linear(100, 100)
        self.activation4 = torch.nn.Sigmoid()
        self.linear5 = torch.nn.Linear(100, 100)
        self.activation5 = torch.nn.Sigmoid()
        self.linear6 = torch.nn.Linear(100, 100)
        self.activation6 = torch.nn.Sigmoid()
        self.linear7 = torch.nn.Linear(100, 100)
        self.activation7 = torch.nn.Sigmoid()
        self.linear8 = torch.nn.Linear(100, 100)
        self.activation8 = torch.nn.Sigmoid()
        self.linear9 = torch.nn.Linear(100, 100)
        self.activation9 = torch.nn.Sigmoid()
        self.linear10 = torch.nn.Linear(100, 20)
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.activation1(O)
        O = self.linear2(O)
        O = self.activation2(O)
        O = self.linear3(O)
        O = self.activation3(O)
        O = self.linear4(O)
        O = self.activation4(O)
        O = self.linear5(O)
        O = self.activation5(O)
        O = self.linear6(O)
        O = self.activation6(O)
        O = self.linear7(O)
        O = self.activation7(O)
        O = self.linear8(O)
        O = self.activation8(O)
        O = self.linear9(O)
        O = self.activation9(O)
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
        
        self.conv1a = MyConv2d(3, 5, kernel_size=5, padding="same")
        self.relu1a = torch.nn.ReLU()
        self.conv1b = MyConv2d(5, 5, kernel_size=5, padding="same")
        self.relu1b = torch.nn.ReLU()
        self.conv1c = MyConv2d(5, 5, kernel_size=5, padding="same")
        self.relu1c = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2a = MyConv2d(5, 10, kernel_size=3, padding="same")
        self.relu2a = torch.nn.ReLU()
        self.conv2b = MyConv2d(10, 10, kernel_size=3, padding="same")
        self.relu2b = torch.nn.ReLU()
        self.conv2c = MyConv2d(10, 10, kernel_size=3, padding="same")
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