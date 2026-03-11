import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from typing import Protocol

SMALL_USERS = 1411
SMALL_ITEMS = 3327

class GreedyPolicy:
    def __init__(
            self,
            greedy_items: np.ndarray,
            available_items: np.ndarray[np.bool]
    ):
        # Store the top items for each user (calculated by presorting the similarity scores)
        self.user_greedy_items = greedy_items

        # For each user, keep track of which indices are next to show
        self.user_next = np.zeros((SMALL_USERS,), dtype=int)

        # Also keep track of what interactions each user is missing
        self.available_items = available_items
    
    def recommend(self, user_id: int) -> int:
        # Recommend the top item that it hasn't shown yet for this user, given that it is available to show
        for i in range(self.user_next[user_id], SMALL_ITEMS):
            item_id = self.user_greedy_items[user_id, i]
            if self.available_items[user_id, item_id]:
                self.user_next[user_id] = i + 1
                return item_id

class Simulator:
    def __init__(
            self,
            small_matrix: KRSmall,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor
    ):
        # Compute a matrix to indicate all interactions that are 'valid' i.e. don't need masking out
        # This is because despite 99.7% density, some interactions are missing
        self.available_interactions = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=np.bool)
        self.available_interactions[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = True

        # Also compute a ground truth reward matrix for all positive user-item interactions
        self.rewards = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=np.bool)
        self.rewards[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = small_matrix.intr_signals

        # Precompute similarity scores for GreedyPolicy; it is wasteful to recompute it each time
        # Use these to derive the item IDs for GreedyPolicy as it has fixed behaviour
        with torch.inference_mode():
            dot_products = user_embeddings @ item_embeddings.T
            self.user_greedy_items = torch.sort(dot_products, dim=1, descending=True).indices.cpu().numpy()

    def run(
            self,
            rounds:int = 100000
    ):
        # Generate the random stream of users
        rng = np.random.default_rng()
        stream = rng.integers(low=0, high=SMALL_USERS, size=rounds)

        # Set up the policies
        greedy = GreedyPolicy(
            self.user_greedy_items,
            self.available_interactions
        )

        greedy_reward = 0
        for round in range(rounds):
            # Retrieve the random user for this round of recommendation
            user_id = stream[round]

            # Simulate the policies, given this user
            greedy_reward += self.rewards[user_id, greedy.recommend(user_id)]
        
        print(f"Over {rounds} rounds, GREEDY achieves cumulative reward: {greedy_reward}")
        print(f"GREEDY accuracy: {greedy_reward / rounds}")