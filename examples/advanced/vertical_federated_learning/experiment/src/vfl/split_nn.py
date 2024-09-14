import torch

import torch.nn as nn


class SplitNN(nn.Module):
    def __init__(self, split_id: int):
        super().__init__()
        
        assert split_id in [0, 1], f"Expected split id to be `0` or `1`, but got {split_id}"
        
        self.split_id = split_id
        
        self.cardx_layer = nn.Sequential(
            nn.Linear(9, 3)
        )
        
        self.scb_layer = nn.Sequential(
            nn.Linear(6, 4)
        )
        
        self.head_layer = nn.Sequential(
            nn.Linear(3 + 4, 1)
        )
        
    def forward(self, x, xs=None):
        if self.split_id == 0:  # SCB
            assert xs is not None, "SCB expected `xs` to have a value"
            x = self.scb_layer(x)
            x = torch.concat([x, xs], axis=1)
            x = self.head_layer(x)
        elif self.split_id == 1:  # CardX
            x = self.cardx_layer(x)
        return x

    def get_split_id(self) -> int:
        return self.split_id


if __name__ == "__main__":
    BATCH_SIZE = 4
    scb_model = SplitNN(0)
    x = torch.ones((BATCH_SIZE, 5))
    xs = torch.ones((BATCH_SIZE, 3))
    
    print(scb_model.forward(x, xs))
    
    cardx_model = SplitNN(1)
    x = torch.ones((BATCH_SIZE, 5))
    
    print(cardx_model.forward(x))
