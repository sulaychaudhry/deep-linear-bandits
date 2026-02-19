import torch
from torch import nn

USER_ID_DIMS = ITEM_ID_DIMS = 64

class UserTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.user_id_embedding = nn.Embedding(7176, USER_ID_DIMS)

        self.user_tower = nn.Sequential(
            nn.Linear(USER_ID_DIMS, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.item_id_embedding = nn.Embedding(10728, ITEM_ID_DIMS)

        self.item_tower = nn.Sequential(
            nn.Linear(ITEM_ID_DIMS, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.user_id

