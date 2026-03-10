import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from typing import Protocol

SMALL_USERS = 1411
SMALL_ITEMS = 3327

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
            user_item_similarities: torch.Tensor
    ):
        # Compute similarities between users and items
        similarities = user_embeddings @ item_embeddings.T

        # Now determine the top items for each user
        similarities = torch.sort(similarities, dim=1).indices

class Simulator:
    @torch.no_grad
    def __init__(
            self,
            small_matrix: KRSmall,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor
    ):
        # Compute matrix for masking out items that a user doesn't interact with
        # The small matrix is highly dense, but some interactions are missing
        available_items = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=np.bool)
        available_items[small_matrix.intr_user_ids, small_matrix.intr_item_ids] = True

        # Compute user-item similarity scores for the greedy policy
        similarities = user_embeddings @ item_embeddings.T

        # Compute matrix necessary for observing rewards;

        # the simulator essentially needs to:
        # 1. pick a random user (assuming that the user hasn't seen everything)
        # 2. pass the user to all of the policies
        # 3. all of the policies need to return their predicted item
        # 4. the simulator needs to observe the reward for that item
        # 5. it then loops