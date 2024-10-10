from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseDataset(ABC):
    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Gibt das Dataset zurück."""
        pass

    @abstractmethod
    def get_dataloader(self, batch_size: int, shuffle: bool = True):
        """Gibt den DataLoader zurück."""
        pass