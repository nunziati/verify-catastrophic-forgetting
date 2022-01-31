import torch

class DatasetMerger(torch.utils.data.Dataset):
    def __init__(self, datasets_list, mode="torch", set_labels_from=None):
        self.datasets_list = datasets_list

        if set_labels_from is None:
            self.class_map = {}
        else:
            self.class_map = set_labels_from.class_map

        data_indexes_list = []

        for dataset_index, dataset in enumerate(self.datasets_list):
            n_samples = len(dataset)
            dataset_index_array = torch.full((1, n_samples), dataset_index, dtype=torch.int64)
            example_index_array = torch.arange(n_samples, dtype=torch.int64).reshape((1, -1))
            data_index_array = torch.cat((dataset_index_array, example_index_array))
            data_indexes_list.append(data_index_array)
            
            if set_labels_from is None:
                self.class_map.update({(dataset_index, x.item()): x.item() + len(self.class_map) for x in torch.unique(torch.tensor(dataset.targets), sorted=True)})
        
        self.data_indexes = torch.cat(data_indexes_list, dim=1).transpose(0, 1)
        self.reverse_class_map = {self.class_map[x]: x for x in self.class_map}

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.data_indexes[index]
        dataset_index = index[0].item()
        example_index = index[1]
        data, label = self.datasets_list[dataset_index][example_index]
        label = self.class_map[(dataset_index, label)]
        return data, label

    def __len__(self):
        return self.data_indexes.shape[0]

    def shuffle(self):
        indexes = torch.randperm(self.data_indexes.shape[0])
        self.data_indexes = self.data_indexes[indexes]

    def dataset_wise_sort_by_label(self):
        # maybe it exists a better way of doing this? maybe without the for loops
        self.data_indexes = self.data_indexes[torch.argsort(self.data_indexes[:, 0]), :]

        data_indexes_list = []

        for dataset_index, dataset in enumerate(self.datasets_list):
            length = len(dataset)
            dataset_index_array = torch.full((1, length), dataset_index, dtype=torch.int64)
            example_index_array = torch.argsort(torch.tensor(dataset.targets)).reshape((1, length))
            data_index_array = torch.cat((dataset_index_array, example_index_array))
            data_indexes_list.append(data_index_array)
        self.data_indexes = torch.cat(data_indexes_list, axis=1).transpose(0, 1)