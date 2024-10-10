from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from proto_grasp.data.base_dataset import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_dataset(self):
        return datasets.MNIST(root=self.root, train=self.train, transform=self.transform, download=self.download)

    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)