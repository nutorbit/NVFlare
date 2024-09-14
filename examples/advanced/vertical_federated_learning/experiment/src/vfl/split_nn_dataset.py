import os

import pandas as pd
import numpy as np

import torch


class SplitNNDataset:
    def __init__(self, root: str, intersect_idx, split_id: int):
        self.root = root
        self.intersect_idx = intersect_idx
        self.split_id = split_id
        self.site_id = f"site-{split_id + 1}"
        
        if self.intersect_idx is not None:
            self.intersect_idx = np.sort(self.intersect_idx).astype(np.int64)
            
        self.data, self.target = self.__build__()
            
    def __build__(self):
        data_path = self.root.replace("site-x", self.site_id)
        
        data_split_dir = os.path.dirname(data_path)
        train_path = os.path.join(data_split_dir, "bank.data.csv")
        # valid_path = os.path.join(data_split_dir, "valid.csv")  # TODO: add validation set
        
        df = pd.read_csv(train_path)
        df = df.set_index('uid')
        
        data = target = None
        if self.split_id == 0:  # SCB
            data, target = df.iloc[self.intersect_idx, 1:].to_numpy(), df.iloc[self.intersect_idx, [0]].to_numpy()
        elif self.split_id == 1:  # CardX
            data = df.iloc[self.intersect_idx].to_numpy()
            
        return data, target
    
    def __len__(self):
        return len(self.data)
    
    def get_batch(self, batch_indices):
        data_batch = self.data[batch_indices]
        if self.split_id == 0: # SCB
            target_batch = self.target[batch_indices]
            return torch.tensor(data_batch, dtype=torch.float32), torch.tensor(target_batch, dtype=torch.float32)
        return torch.tensor(data_batch, dtype=torch.float32)


if __name__ == "__main__":
    dataset = SplitNNDataset(
        root="/tmp/nvflare/vertical_data/site-x/",
        intersect_idx=list(range(100)),
        split_id=1
    )
    
    print(dataset.get_batch(list(range(10))))