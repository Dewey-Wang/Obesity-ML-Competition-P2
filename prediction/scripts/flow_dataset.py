import torch
from torch.utils.data import Dataset
import numpy as np

class FlowPairDataset(Dataset):

    def __init__(self, pairs):

        self.z0 = torch.tensor(
            np.stack([p["z0"] for p in pairs]),
            dtype=torch.float32
        )

        self.z1 = torch.tensor(
            np.stack([p["z1"] for p in pairs]),
            dtype=torch.float32
        )

        self.cond = torch.tensor(
            np.stack([p["cond"] for p in pairs]),
            dtype=torch.float32
        )


    def __len__(self):

        return len(self.z0)


    def __getitem__(self, idx):

        return (
            self.z0[idx],
            self.z1[idx],
            self.cond[idx]
        )
        