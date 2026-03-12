import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from typing import Protocol
from tqdm import tqdm

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

class LinUCB:
    def __init__(
            self,
            device: torch.device,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor,
            available_items: torch.Tensor,
            alpha: float = 1.0,  # exploration bonus
            lmbda: float = 1.0   # ridge regularisation penalty (higher = more weight decay)
    ):
        self.device = device

        # Store user & item embeddings + alpha (exploration bonus)
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.alpha = alpha

        # available_items needs copying so that it can be modified (removing seen items after serving them)
        self.available_items = available_items.detach().clone()

        # Calculate the size of the context vector: the user embeddings, item embeddings, then their element-wise product
        context_size = user_embeddings.shape[1] * 3
        
        # A_t = lambda * I + (sum of all prior context vector self-dot-products i.e. their squared L2 norm)
        # therefore the inverse starts as (1 / lambda) * I
        self.A_inv = (1 / lmbda) * torch.eye(context_size, device=self.device)

        # b_t = all reward-weighted (via dot product) context vectors seen so far; therefore initialised at 0
        self.b = torch.zeros(context_size, device=self.device)
    
    def _compute_contexts(self, user_id: int):
        # Look up this user's user embedding vector (will have size D = tower output dimensions)
        user_emb = self.user_embeddings[user_id, :]

        # Item embeddings are already known
        item_embs = self.item_embeddings

        # User vector: (D,)
        # Item vector: (3327, D)
        # Desired output (user vector concatenated with item vector & user-item element-wise multiplication): (3327, D*3)
        # Therefore will need to broadcast the user vector for concatenation
        user_emb = user_emb.unsqueeze(0) # (1, D) so can now be broadcasted across all items

        # Now compute context vectors via broadcasting; the user vector is repeated along all items
        # Result will have size (3327, D*3) = (3327, context_size) as needed
        return torch.cat([
            user_emb.expand_as(item_embs),  # (1, D) -> (3327, D) broadcasted
            item_embs, # (3327, D)
            user_emb * item_embs # (1, D) * (3327, D) -> (3327, D) * (3327, D) broadcasted
        ], dim=1)

    def recommend(self, user_id: int) -> int:
        # Compute Theta^ = A_inv @ b where Theta is the current best estimate of the linear context weights
        theta_hat = self.A_inv @ self.b # (context_size,)

        # Compute all context vectors for this user (this user across all items)
        contexts = self._compute_contexts(user_id) # (3327, context_size)

        # Reward estimates are just contexts @ theta_hat
        reward_estimates = contexts @ theta_hat

        # Now compute exploration bonuses
        # For a single context vector c in contexts, the exploration bonus is:
        #           alpha * sqrt(c^T @ A_inv @ c)
        # i.e. each context gets matmul with all rows/cols of A_inv (symmetric so doesn't matter) then dot product with self is computed
        #      then the square root
        # This can be done for the entire matrix via:
        # - compute (contexts @ A_inv) to apply A_inv across each context vector to get (3327, context_size) result
        # - then simply dot product all context rows with self manually
        X = contexts @ self.A_inv
        exploration = self.alpha * torch.sqrt(
            (X * contexts).sum(dim=1)
        )

        # UCB scores are just reward estimates + exploration bonuses
        scores = reward_estimates + exploration

        # Now mask out unavailable (no interaction or already shown) items
        scores[~self.available_items[user_id]] = -torch.inf

        # Pick item with highest UCB score
        item_id = scores.argmax().item()

        # Mark chosen item as unavailable now too
        self.available_items[user_id, item_id] = False

        return item_id
    
    def update(self, user_id:int, item_id:int, reward:bool):
        # Get feature vector (psi) for this specific user-item pair
        u = self.user_embeddings[user_id]
        i = self.item_embeddings[item_id]
        psi = torch.cat([u, i, u * i])

        # Now update A & b:
        # - A_(t+1) = A_t + dotprod(psi_t, psi_t)
        # - b_(t+1) = b_t + reward_t * psi_t

        # Note that A_inv is needed though, so computationally expensive to do this each time
        # However the update is a rank-1 update so can use Sherman-Morrison directly on A_inv
        # (A + uv^T)_inv = A_inv - (A_inv u v^T A_inv)/(1 + v^T A_inv u)
        X = self.A_inv @ psi
        denom = 1 + psi.T @ X
        self.A_inv -= torch.outer(X, X) / denom

        # b is much easier to update
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

        # Compute a matrix to indicate all interactions that are 'valid' i.e. don't need masking out
        # This is because despite 99.7% density, some interactions are missing
        self.available_interactions = torch.zeros(SMALL_USERS, SMALL_ITEMS, dtype=torch.bool, device=device)
        self.available_interactions[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = True

        # Also compute a ground truth reward matrix for all positive user-item interactions
        self.rewards = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=np.bool)
        self.rewards[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = small_matrix.intr_signals

        # Precompute similarity scores for GreedyPolicy; it is wasteful to recompute it each time
        # Use these to derive the item IDs for GreedyPolicy as it has fixed behaviour
        dot_products = user_embeddings @ item_embeddings.T
        self.user_greedy_items = torch.sort(dot_products, dim=1, descending=True).indices.cpu().numpy()

        # Store user & item embeddings
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

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
            self.available_interactions.cpu().numpy()
        )
        linucb = LinUCB(
            self.device,
            self.user_embeddings,
            self.item_embeddings,
            self.available_interactions
        )

        greedy_c_reward = 0
        linucb_c_reward = 0
        for round in tqdm(range(rounds)):
            # Retrieve the random user for this round of recommendation
            user_id = stream[round]

            # Simulate greedy policy
            greedy_c_reward += self.rewards[user_id, greedy.recommend(user_id)]

            # Simulate LinUCB
            linucb_item = linucb.recommend(user_id)
            linucb_reward = self.rewards[user_id, linucb_item]
            linucb.update(user_id, linucb_item, linucb_reward)
            linucb_c_reward += linucb_reward

        print(f"Over {rounds} rounds, GREEDY achieves cumulative reward: {greedy_c_reward}")
        print(f"Over {rounds} rounds, LinUCB achieves cumulative reward: {linucb_c_reward}")