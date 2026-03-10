import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from typing import Protocol

# user & item embeddings are for the contextual bandits & also for the greedy policy
# policies are just classes that the simulator uses to pick what to do with a user that it simulates entering the environment
# the states are tracked by the policies; it's up to the simulator to just set up those policies correctly here

# so it has the small matrix so that it can observe rewards at each step, but also so that it can set up the policies

class Policy(Protocol):
    def recommend(user_id: int) -> int: ...

class GreedyPolicy(Protocol):
    # The greedy policy just aims to give each user the item with highest similarity

    @torch.no_grad
    def __init__(
            self,
            small_matrix: KRSmall,
            user_embeddings: torch.Tensor, # (1411, 64)
            item_embeddings: torch.Tensor  # (3327, 64)
    ):
        # Compute similarities between users and items
        similarities = user_embeddings @ item_embeddings.T

        # Now determine the top items for each user
        similarities = torch.sort(similarities, dim=1).indices

class Simulator:
    def __init__(
            self,
            small_matrix: KRSmall,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor
    ):
        # The linear contextual bandits need a specific feature map of the user & item embeddings
        # These need precomputing for each user-item pair; the feature map is specifically:
        #       concat( user embedding, item embedding, element-wise user-item embedding product )
        u = user_embeddings.unsqueeze(1) # (1411, 1, 64)
        i = item_embeddings.unsqueeze(0) # (1, 3327, 64)
        bandit_contexts = torch.cat(
            u.expand(-1, i.shape[1], -1), # -> (1411, 3327, 64) i.e. broadcast over all items
            i.expand(u.shape[0], -1, -1), # -> (1411, 3327, 64) i.e. broadcast over all users
            u * i                         # Automatically broadcasted due to matching 64
        )