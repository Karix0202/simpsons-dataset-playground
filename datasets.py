from torch.utils.data import Dataset

class Subset(Dataset):
    def __init__(self, indices, parent_dataset):
        self.indices = indices
        self.parent_dataset = parent_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return parent_dataset[self.indices]
