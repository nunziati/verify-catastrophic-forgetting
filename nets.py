import torch

class MyConv2d(torch.nn.Conv2d):
    """Custom version of torch.nn.Conv2d, built because paddint="same" is not a valid option in some versions of pytorch.
        There is no need of a forward method, because torch.nn.Conv2d already defines it."""

    def __init__(self, in_maps, out_maps, *args, padding="same", **kwargs):
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
        super(MyConv2d, self).__init__(in_maps, out_maps, *args, padding=padding, **kwargs)

class ShallowMLP(torch.nn.Module):
    def __init__(self, hidden_units=100, dropout=0.0):
        super(ShallowMLP, self).__init__()

        self.register_buffer("hidden_units", torch.LongTensor([hidden_units]))
        self.register_buffer("droput", torch.LongTensor([dropout]))

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, hidden_units)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_units, 20)
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
    def __init__(self, hidden_units=100, dropout=0.0):
        super(DeepMLP, self).__init__()

        self.register_buffer("hidden_units", torch.LongTensor([hidden_units]))
        self.register_buffer("droput", torch.LongTensor([dropout]))

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(28*28*3, hidden_units)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation2 = torch.nn.Sigmoid()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.linear3 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation3 = torch.nn.Sigmoid()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.linear4 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation4 = torch.nn.Sigmoid()
        self.dropout4 = torch.nn.Dropout(dropout)
        self.linear5 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation5 = torch.nn.Sigmoid()
        self.dropout5 = torch.nn.Dropout(dropout)
        self.linear6 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation6 = torch.nn.Sigmoid()
        self.dropout6 = torch.nn.Dropout(dropout)
        self.linear7 = torch.nn.Linear(hidden_units, hidden_units)
        self.activation7 = torch.nn.Sigmoid()
        self.dropout7 = torch.nn.Dropout(dropout)
        self.linear8 = torch.nn.Linear(hidden_units, 20)
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        O = self.flatten(X)
        O = self.linear1(O)
        O = self.activation1(O)
        O = self.dropout1(O)
        O = self.linear2(O)
        O = self.activation2(O)
        O = self.dropout2(O)
        O = self.linear3(O)
        O = self.activation3(O)
        O = self.dropout3(O)
        O = self.linear4(O)
        O = self.activation4(O)
        O = self.dropout4(O)
        O = self.linear5(O)
        O = self.activation5(O)
        O = self.dropout5(O)
        O = self.linear6(O)
        O = self.activation6(O)
        O = self.dropout6(O)
        O = self.linear7(O)
        O = self.activation7(O)
        O = self.dropout7(O)
        A = self.linear8(O)
        O = self.softmax(A)

        return O, A
    
    def initialize(self):
        for param in self.parameters():
            torch.nn.init.normal_(param)

class ShallowCNN(torch.nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        
        self.net_parameters = {}

        self.conv1 = MyConv2d(3, 10, kernel_size=5, padding="same")
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = MyConv2d(10, 20, kernel_size=3, padding="same")
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7*7*20, 20)
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
        
        self.net_parameters = {}

        self.conv1a = MyConv2d(3, 10, kernel_size=5, padding="same")
        self.relu1a = torch.nn.ReLU()
        self.conv1b = MyConv2d(10, 10, kernel_size=5, padding="same")
        self.relu1b = torch.nn.ReLU()
        self.conv1c = MyConv2d(10, 10, kernel_size=5, padding="same")
        self.relu1c = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2a = MyConv2d(10, 20, kernel_size=3, padding="same")
        self.relu2a = torch.nn.ReLU()
        self.conv2b = MyConv2d(20, 20, kernel_size=3, padding="same")
        self.relu2b = torch.nn.ReLU()
        self.conv2c = MyConv2d(20, 20, kernel_size=3, padding="same")
        self.relu2c = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(7*7*20, 20)
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