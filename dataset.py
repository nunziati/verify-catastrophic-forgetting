import torch

class DatasetMerger(torch.utils.data.Dataset):
    """Class that implements a dataset that is able to merge more datasets in one in a transparent way."""
    
    def __init__(self, datasets_list, set_labels_from=None):
        """Initialize the dataset.

        Args:
            datasets_list: an iterable of datasets: the datasets that we want to merge.
            set_labels_from: an instance of this class, that is used to create the labels in a consistent way; if None, the labels are set automatically.
        """

        self.datasets_list = list(datasets_list)
        self.class_map = {} # to map the label unique identifier with the couple (dataset_id, label_within_the_dataset)
        self.reverse_class_map = {}
        data_indexes_list = []

        # copying the labels from another dataset (e.g. useful when creating a test set, and want to keep the same labels of a training set)
        if set_labels_from is not None:
            self.class_map = set_labels_from.class_map

        # building a 2D tensor with columns (dataset_id, example_id_whithin_the_dataset)
        for dataset_index, dataset in enumerate(self.datasets_list):
            n_samples = len(dataset)

            # creating the two columns for the considered dataset
            dataset_index_array = torch.full((1, n_samples), dataset_index, dtype=torch.int64)
            example_index_array = torch.arange(n_samples, dtype=torch.int64).reshape((1, -1))

            # concatenating the two columns
            data_index_array = torch.cat((dataset_index_array, example_index_array))
            
            # preparing the columns to be concatenated with other columns
            data_indexes_list.append(data_index_array)
            
            # extract the labels of each dataset and give them a unique id
            if set_labels_from is None:
                self.class_map.update({(dataset_index, x.item()): x.item() + len(self.class_map) for x in torch.unique(torch.LongTensor(dataset.targets), sorted=True)})
        
        # concatenating the tables of each dataset
        self.data_indexes = torch.cat(data_indexes_list, dim=1).transpose(0, 1)

        self.reverse_class_map = {self.class_map[x]: x for x in self.class_map}
        self.num_classes = len(self.class_map)

    def __getitem__(self, index):
        """Get an example from the dataset.
        
        Args:
            index: can be an int or a couple (dataset_id, example_id_whithin_the_dataset).
        Returns:
            The pair (pattern, label)
        """

        if isinstance(index, int):
            index = self.data_indexes[index]

        dataset_index = index[0].item()
        example_index = index[1]
        data, label = self.datasets_list[dataset_index][example_index]
        label = self.class_map[(dataset_index, label)] # uses the unique identifier of the label
        return data, label

    def __len__(self):
        """Return the total number of examples in this dataset."""

        return self.data_indexes.shape[0]

    def shuffle(self):
        """Shuffle the examples of the dataset, mixing togeher examples of different datasets."""

        indexes = torch.randperm(self.data_indexes.shape[0])
        self.data_indexes = self.data_indexes[indexes]

    def dataset_wise_sort_by_label(self):
        """Sort the dataset by target class id."""

        data_indexes_list = []

        # build again the dataset, but sorted by label
        for dataset_index, dataset in enumerate(self.datasets_list):
            length = len(dataset)

            # creating the two columns for the considered dataset
            dataset_index_array = torch.full((1, length), dataset_index, dtype=torch.int64)
            example_index_array = torch.argsort(torch.LongTensor(dataset.targets)).reshape((1, length))

            # concatenating the two columns
            data_index_array = torch.cat((dataset_index_array, example_index_array))

            # preparing the columns to be concatenated with other columns
            data_indexes_list.append(data_index_array)

        # concatenating the tables of each dataset
        self.data_indexes = torch.cat(data_indexes_list, axis=1).transpose(0, 1)