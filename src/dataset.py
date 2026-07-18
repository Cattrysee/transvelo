import torch
from torch.utils.data import Dataset

class VelocityDataset(Dataset):

    def __init__(self, spliced, unspliced, condition):

        self.S = spliced
        self.U = unspliced
        self.cond = condition

    def __len__(self):

        return self.S.shape[0]

    def __getitem__(self, idx):

        return {
            "S": self.S[idx],
            "U": self.U[idx],
            "cond": self.cond[idx],  # restored to 'mixed-data' mode; for 'separate-data' mode, set cond to None at the call site
        }
