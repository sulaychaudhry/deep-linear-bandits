import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from tqdm import trange

SMALL_USERS = 1411
SMALL_ITEMS = 3327

class GreedyPolicy:
    def __init__(
            self,
            greedy_items: np.ndarray,
            available: np.ndarray[np.bool]
    ):
        # Store the top items for each user (calculated by presorting the similarity scores)
        self.greedy_items = greedy_items

        # For each user, keep track of which indices are next to show
        self.user_next = np.zeros((SMALL_USERS,), dtype=int)

        # Also keep track of what interactions each user is missing
        self.available = available

    def recommend(self, user_id: int) -> int:
        # Recommend the top item that it hasn't shown yet for this user, given that it is available to show
        for i in range(self.user_next[user_id], SMALL_ITEMS):
            item_id = self.greedy_items[user_id, i]
            if self.available[user_id, item_id]:
                self.user_next[user_id] = i + 1
                return item_id

class LinUCB:
    def __init__(
            self,
            device: torch.device,
            contexts: torch.Tensor,    # (1411, 3327, D)
            available: torch.Tensor,   # (1411, 3327)

            alpha: float = 1.0,        # Exploration bonus
            lam: float = 1.0           # Ridge regularisation penalty (weight decay)
    ):
        self.device = device
        self.contexts = contexts
        self.available = available.detach().clone() # Needs copying so that it can be modified
        self.alpha = alpha

        # Set up A_inv, which starts as (1 / lambda) * I
        self.A_inv = (1 / lam) * torch.eye(contexts.shape[-1], device=self.device) # (D, D)

        # Set up b, which starts as 0
        self.b = torch.zeros(contexts.shape[-1], device=self.device)               # (D,)

    def recommend(self, user_id: int) -> int:
        # Compute (theta = A_inv @ b) where theta is the current best estimate of the linear weights
        theta = self.A_inv @ self.b # (D,)

        # Retrieve the context vectors (Psi) for this user
        Psi = self.contexts[user_id] # (3327, D)

        # Reward estimates are (Psi @ theta) to apply weights across all contexts
        rewards = Psi @ theta

        # Now compute exploration bonuses, which for a single context vector c is:
        #       alpha * sqrt(c^T @ A_inv @ c)
        # i.e. each c has A_inv applied to it, then dot producted with the original c; before sqrt and alpha
        C = Psi @ self.A_inv
        exploration = self.alpha * torch.sqrt(
            (C * Psi).sum(dim=1)
        )

        # UCB scores are just reward estimates + exploration bonuses
        scores = rewards + exploration

        # Now mask out unavailable (no interaction or already shown) items
        scores[~self.available[user_id]] = -torch.inf

        # Pick item with highest UCB score
        item_id = scores.argmax().item()

        # Mark chosen item as unavailable now too
        self.available[user_id, item_id] = False

        return item_id

    def update(
            self,
            user_id: int,
            item_id: int,
            reward: bool
    ):
        # Get feature vector (psi) for this specific user-item pair
        psi = self.contexts[user_id, item_id] # (D,)

        # Update A_inv using Sherman-Morrison
        # (A + psi psi^T)_inv = A_inv - (A_inv psi psi^T A_inv) / (1 + psi^T A_inv psi)
        #                     = A_inv - (outer(A_inv psi, A_inv psi)) / (1 + psi^T A_inv psi)
        C = self.A_inv @ psi
        self.A_inv -= torch.outer(C, C) / (1 + psi.t() @ C)

        # Update b trivially
        self.b += reward * psi

class Simulator:
    @torch.inference_mode()
    def __init__(
            self,
            device: torch.device,
            small_matrix: KRSmall,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor
    ):
        self.device = device

        # Precompute all context vectors across all users and all items; this is large (over 3GB) but will fit in VRAM
        u = user_embeddings.unsqueeze(1)  # (1411, 1, 64)
        i = item_embeddings.unsqueeze(0)  # (1, 3327, 64)
        self.contexts = torch.cat([
                u.expand(-1, SMALL_ITEMS, -1), # -> (1411, 3327, 64)
                i.expand(SMALL_USERS, -1, -1), # -> (1411, 3327, 64)
                u * i                          # -> (1411, 3327, 64) since auto broadcasts
        ], dim=-1)                             # =  (1411, 3327, D) where D=192

        # Compute a matrix to indicate all interactions that are 'valid' i.e. don't need masking out
        # This is because despite 99.7% density, some interactions are missing
        self.available = torch.zeros(SMALL_USERS, SMALL_ITEMS, dtype=torch.bool, device=device)
        self.available[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = True

        # Also compute a ground truth reward matrix for all positive user-item interactions
        self.rewards = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=np.bool)
        self.rewards[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = small_matrix.intr_signals

        # Precompute similarity scores for GreedyPolicy
        # This can be used immediately to derive the item IDs for GreedyPolicy as it has fixed behaviour
        dot_products = user_embeddings @ item_embeddings.T
        self.greedy_items = torch.sort(dot_products, dim=1, descending=True).indices.cpu().numpy()
        
    def run(
            self,
            rounds:int = 100000
    ):
        # Generate the random stream of users
        rng = np.random.default_rng()
        stream = rng.integers(low=0, high=SMALL_USERS, size=rounds)

        # Set up policies
        greedy = GreedyPolicy(
            self.greedy_items,
            self.available.cpu().numpy()
        )
        linucb = LinUCB(
            self.device,
            self.contexts,
            self.available
        )

        # Simulate rounds
        linucb_rewards = 0
        greedy_rewards = 0
        for round in trange(rounds, desc="Simulation rounds"):
            # Retrieve the random user for this round
            user_id = stream[round]

            # Simulate GreedyPolicy & update reward
            greedy_item_rec = greedy.recommend(user_id)
            greedy_rewards += self.rewards[user_id, greedy_item_rec]

            # Simulate LinUCB
            linucb_item_rec = linucb.recommend(user_id)
            linucb_reward = self.rewards[user_id, linucb_item_rec]

            # Update LinUCB cumulative reward & model
            linucb_rewards += linucb_reward
            linucb.update(user_id, linucb_item_rec, linucb_reward)
        
        print(f"Cumulative GreedyPolicy reward over {rounds} rounds: {greedy_rewards} ({greedy_rewards/rounds})")
        print(f"Cumulative LinUCB reward over {rounds} rounds: {linucb_rewards} ({linucb_rewards/rounds})")