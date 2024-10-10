from proto_grasp.data.mnist_dataset import MNISTDataset

class DatasetFactory:
    def __init__(self):
        self.datasets = {
            'mnist': MNISTDataset
        }

    def get_dataset(self, dataset_name: str):
        dataset_class = self.datasets.get(dataset_name.lower())
        if dataset_class is None:
            raise ValueError(f"Datensatz {dataset_name} ist nicht verfügbar.")
        return dataset_class()