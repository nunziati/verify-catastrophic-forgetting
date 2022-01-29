import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def interactive():
    while(True):
        a = input(">>> ")
        if a=="q": break
        eval(a)

def show_example(a):
    plt.imshow(a[0].squeeze())
    plt.show()

class DatasetMerger(torch.utils.data.Dataset):
    def __init__(self, datasets_list, mode="torch"):
        self.datasets_list = datasets_list
        indexes_list = []
        self.class_map = {}
        for dataset_index, dataset in enumerate(self.datasets_list):
            dim = len(dataset)
            dataset_index_array = np.full(dim, dataset_index, dtype=np.int64).reshape((1, dim))
            example_index_array = np.arange(dim, dtype=np.int64).reshape((1, dim))
            array = np.concatenate((dataset_index_array, example_index_array))
            indexes_list.append(array)
            self.class_map.update({(dataset_index, x): x + len(self.class_map) for x in np.sort(np.unique(dataset.targets))})
        self.data_indexes = np.concatenate(indexes_list, axis=1).transpose()
        self.reverse_class_map = {self.class_map[x]: x for x in self.class_map}
        
            

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.data_indexes[index]
        return index, self.datasets_list[index[0]][index[1]]

    def shuffle(self):
        self.data_list = np.random.shuffle(self.data_indexes)

    def dataset_wise_sort_by_label(self):
        # maybe it exists a better way of doing this? maybe without the for loops
        self.data_indexes = self.data_indexes[np.argsort(self.data_indexes[:, 0]), :]

        data_indexes_list = []

        for dataset_index, dataset in enumerate(self.datasets_list):
            dim = len(dataset)
            dataset_index_array = np.full(dim, dataset_index, dtype=np.int64).reshape((1, dim))
            example_index_array = np.argsort(dataset.targets).reshape((1, dim))
            array = np.concatenate((dataset_index_array, example_index_array))
            data_indexes_list.append(array)
        self.data_indexes = np.concatenate(data_indexes_list, axis=1).transpose()

class ShallowMLP(torch.nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(784, 100)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(100, 100)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.linear3 = torch.nn.Linear(100, 10)
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

    
#if __name__ == "__main__":
device = torch.device("cpu")

print("Downloading datasets...")
transform_cifar10 = transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Lambda(lambda pic: torch.from_numpy(pic.numpy().transpose((1, 2, 0))))])
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda pic: torch.from_numpy(pic.repeat(3, 1, 1).numpy().transpose((1, 2, 0))))])

train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
train_set_mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_set_mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
print("Datasets correctly downloaded")

print("Loading data...")
"""train_loader_cifar10 = torch.utils.data.DataLoader(train_set_cifar10, batch_size=1, shuffle=False, num_workers=2)
test_loader_cifar10 = torch.utils.data.DataLoader(test_set_cifar10, batch_size=4, shuffle=False, num_workers=2)
train_loader_mnist = torch.utils.data.DataLoader(train_set_mnist, batch_size=1, shuffle=False, num_workers=2)
test_loader_mnist = torch.utils.data.DataLoader(test_set_mnist, batch_size=4, shuffle=False, num_workers=2)
"""
print("Dataset correctly loaded!")

train_set = DatasetMerger((train_set_cifar10, train_set_mnist))

train_set.shuffle()

train_set.dataset_wise_sort_by_label()

for index, data in train_set:
    print(index, "\t", data[1])

print("Sorting and merging datasets...")

# TO DO: sort_dataset(cifar10_train)
# TO DO: sort_dataset(mnist_train)
# TO DO: merge_dataset(cifar10_train, mnist_train)
# TO DO: merge_dataset(cifar10_test, mnist_test) # no need for sorting them
print("Training set correctly sorted!")

print("Creating models...")
"""net = ShallowMLP()
net.to(torch.float32).to(device)
net.initialize()"""
print("Models correctly created!")

print("Starting the training procedure...")
# for each neural model: train()
print("Training correctly completed!")

# plot the result

