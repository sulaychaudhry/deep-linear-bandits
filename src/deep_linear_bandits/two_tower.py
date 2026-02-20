import torch
from torch import nn
import torch.nn.functional as F

USER_ID_DIMS = ITEM_ID_DIMS = 64

"""
Sets up the user tower:
- Embeds sparse user_id to a learned 64-wide dense vector
- Uses the tower set up from the system design
"""
class UserTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.id_embedder = nn.Embedding(7176, USER_ID_DIMS)

        # Tower temporarily smaller for ID-only testing
        self.tower = nn.Sequential(
            nn.Linear(USER_ID_DIMS, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, user_ids):
        return self.tower(
            self.id_embedder(user_ids)
        )

"""
Sets up the item tower:
- Embeds sparse item_id to a learned 64-wide dense vector
- Uses the tower set up from the system design
"""
class ItemTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.id_embedder = nn.Embedding(10728, ITEM_ID_DIMS)

        # Tower temporarily smaller for ID-only testing
        self.tower = nn.Sequential(
            nn.Linear(ITEM_ID_DIMS, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, item_ids):
        return self.tower(
            self.id_embedder(item_ids)
        )

"""
Sets up the Two Tower architecture:
- Passes user_ids through the user tower
- Passes item_ids through the item tower
- Calculates dot product between all users and all items
--- These logits will be converted to probabilities via softmax during CrossEntropyLoss used in training
--- The training will aim to increase the dot product for the known liked item in the batch, decrease the rest for that user
"""
class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()

        self.user_tower = UserTower()
        self.item_tower = ItemTower()
    
    def forward(self, user_ids, item_ids):
        u = self.user_tower(user_ids)
        i = self.item_tower(item_ids)

        u = F.normalize(u)
        i = F.normalize(u)

        # Calculate dot products between users & items
        # These are the raw scores (logits)
        return u @ i.T

