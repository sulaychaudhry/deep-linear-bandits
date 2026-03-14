import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime

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
    
    def update(self, user_id: int, item_id: int, reward: int): pass

class RandomPolicy:
    def __init__(
            self,
            available: torch.Tensor,    # (1411, 3327, D)
            rng: np.random.BitGenerator # RNG instance
    ):
        # Save available items & the RNG
        self.available = available.detach().clone()
        self.rng = rng
    
    def recommend(self, user_id:int) -> int:
        available = self.available[user_id]

        # Pick a random available item, using flatnonzero to efficiently get item IDs of available items
        available = np.flatnonzero(available.cpu().numpy())
        item_id = self.rng.choice(available)

        # Mark chosen item as unavailable for next time this user is in the stream
        self.available[user_id, item_id] = False

        return item_id
    
    def update(self, user_id:int, item_id:int, reward:int): pass

class LinUCB:
    def __init__(
            self,
            device: torch.device,
            contexts: torch.Tensor,    # (1411, 3327, D)
            available: torch.Tensor,   # (1411, 3327)

            alpha: float = 0.5,        # Exploration bonus
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

class EpsilonGreedy:
    def __init__(
            self,
            device: torch.device,
            contexts: torch.Tensor,         # (1411, 3327, D)
            available: torch.Tensor,        # (1411, 3327)

            rng: np.random.BitGenerator,    # RNG instance

            epsilon: float = 0.1,           # Random exploration probability
            lam: float = 1.0                # Ridge regularisation penalty (weight decay)
    ):
        self.device = device
        self.contexts = contexts
        self.available = available.detach().clone() # Needs copying so that it can be modified
        self.epsilon = epsilon
        self.rng = rng

        # Set up posteriors (same as LinUCB, TS)
        self.A_inv = (1 / lam) * torch.eye(contexts.shape[-1], device=self.device) # (D, D)
        self.b = torch.zeros(contexts.shape[-1], device=self.device)               # (D,)

    def recommend(self, user_id:int) -> int:
        available = self.available[user_id]

        # epsilon% of the time, pick a random available item
        if self.rng.random() < self.epsilon:
            # Exploration: pick uniformly from available items

            # Use flatnonzero to efficiently get item IDs (indices) of available items
            available = np.flatnonzero(available.cpu().numpy())
            item_id = self.rng.choice(available)
        else:
            # Exploitation: pick item with highest estimated reward (wrt the linear regression)

            # Calculate weight estimates & contexts; then apply weights to all contexts to get scores
            theta = self.A_inv @ self.b # (D,)
            Psi = self.contexts[user_id] # (3327, D)
            scores = Psi @ theta

            # Masking out unavailable items before picking one
            scores[~available] = -torch.inf

            item_id = scores.argmax().item()

        # Mark chosen item as unavailable
        self.available[user_id, item_id] = False

        return item_id
    
    def update(self, user_id:int, item_id:int, reward:int):
        # Same as LinUCB: update A_inv and b

        # Retrieve context vector for this user-item pair
        psi = self.contexts[user_id, item_id] # (D,)

        # Update A_inv using Sherman-Morrison
        C = self.A_inv @ psi
        self.A_inv -= torch.outer(C, C) / (1 + psi.t() @ C)

        # Update b trivially
        self.b += reward * psi

class ThompsonSampling:
    def __init__(
            self,
            device: torch.device,
            contexts: torch.Tensor,    # (1411, 3327, D)
            available: torch.Tensor,   # (1411, 3327)

            v: float = 1.0,            # Posterior variance scale (higher increases distribution spread, exploration)
            lam: float = 1.0
    ):
        self.device = device
        self.contexts = contexts
        self.available = available.detach().clone() # Needs copying so that it can be modified
        self.v = v

        # Set up posteriors (same as LinUCB, EpsilonGreedy)
        self.A_inv = (1 / lam) * torch.eye(contexts.shape[-1], device=self.device) # (D, D)
        self.b = torch.zeros(contexts.shape[-1], device=self.device)               # (D,)
    
    def recommend(self, user_id: int) -> int:
        # Compute (theta = A_inv @ b) where theta is the current best estimate of the linear weights
        theta = self.A_inv @ self.b # (D,)

        # Retrieve the context vectors (Psi) for this user
        Psi = self.contexts[user_id] # (3327, D)

        # Thompson Sampling samples its ts_theta from:
        #       ts_theta ~ N(theta, v^2 A_inv)
        # where N is a multivariate normal distribution
        # This is efficiently done in practice through Cholesky decomposition to transform standard normal samples
        #       ts_theta = theta + Lz       where z ~ N(0, 1)
        #                                   & L satisfies L L^T = v^2 A_inv
        # L can be computed through torch.linalg.cholesky
        # Additionally v^2 can be factored out, giving:
        #       ts_theta = theta + v * Lz
        try:
            L = torch.linalg.cholesky(self.A_inv)
        except:
            # Sometimes floating-point imprecision means that that the matrix stops being positive-definite
            # This is corrected with adding a small value along the diagonal
            L = torch.linalg.cholesky(
                self.A_inv + 1e-6 * torch.eye(self.A_inv.shape[0], device=self.device)
            )
        z = torch.randn(self.A_inv.shape[0], device=self.device)
        theta += self.v * (L @ z)

        # Now score all contexts using the sampled linear parameters
        scores = Psi @ theta

        # Mask out unavailable items
        scores[~self.available[user_id]] = -torch.inf

        # Pick item with highest score
        item_id = scores.argmax().item()

        # Mark chosen item as unavailable
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
        self.rewards = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=bool)
        self.rewards[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = small_matrix.intr_signals

        # Precompute similarity scores for GreedyPolicy
        # This can be used immediately to derive the item IDs for GreedyPolicy as it has fixed behaviour
        dot_products = user_embeddings @ item_embeddings.T
        self.greedy_items = torch.sort(dot_products, dim=1, descending=True).indices.cpu().numpy()
    
    def _visualise(
            self,
            seed_count: int,
            labels: list[str],
            mean_rewards: np.ndarray[float]
    ):
        # Get cumulative rewards (from the means)
        cum_reward_means = np.cumsum(mean_rewards, axis=1)
        rounds = np.arange(1, mean_rewards.shape[1] + 1)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"Policy simulation: cumulative bandit rewards (averaged over {seed_count} simulations)")

        # Set up all three subplots
        for ax in (ax1, ax2, ax3):
            ax.plot(rounds, cum_reward_means[0], color='#444444', label="Greedy", linestyle='--')
            ax.plot(rounds, cum_reward_means[1], color='#aaaaaa', label="Random", linestyle='--')
            ax.set_xlabel("Round")
            ax.set_ylabel("Cumulative reward")
            ax.grid(True, alpha=0.3)

        # Plot ε-greedy rewards
        j, colours = 0, ['#c6d9f7', '#7aaede', '#4878CF', '#1e3f7a']
        for i, label in enumerate(labels):
            if not label.startswith("ε-greedy"): continue
            ax1.plot(rounds, cum_reward_means[i], color=colours[j], label=label)
            ax1.set_title("ε-greedy rewards")
            ax1.legend()
            j += 1

        # Plot LinUCB rewards
        j, colours = 0, ['#fddbb4', '#f5a962', '#E8612C', '#a83a10', '#5c1a04']
        for i, label in enumerate(labels):
            if not label.startswith("LinUCB"): continue
            ax2.plot(rounds, cum_reward_means[i], color=colours[j], label=label)
            ax2.set_title("LinUCB")
            ax2.legend()
            j += 1

        # Plot Thompson Sampling rewards
        j, colours = 0, ['#a8d8a8', '#6AAB6A', '#3d7a3d', '#1e4f1e', '#0a2a0a']
        for i, label in enumerate(labels):
            if not label.startswith("TS"): continue
            ax3.plot(rounds, cum_reward_means[i], color=colours[j], label=label)
            ax3.set_title("Thompson Sampling")
            ax3.legend()
            j += 1

        plt.tight_layout()
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(f'metrics/bandit/metrics_{timestamp}.png')
        plt.show()
    
    def _run_one_seed(
            self,
            simulation_num: int,
            stream: np.ndarray[int],
            e_greedy_epsilons: list[float],
            linucb_alphas: list[float],
            ts_vs: list[float],
            rounds: int
    ) -> np.ndarray[bool]:
        rng = np.random.default_rng()

        # Set up policies
        policies = {
            "Greedy": GreedyPolicy(self.greedy_items, self.available.cpu().numpy()),
            "Random": RandomPolicy(self.available, rng),
        }
        for eps in e_greedy_epsilons:
            policies[f"ε-greedy (ε={eps})"] = EpsilonGreedy(
                self.device, self.contexts, self.available, rng, epsilon=eps
            )
        for alpha in linucb_alphas:
            policies[f"LinUCB (α={alpha})"] = LinUCB(
                self.device, self.contexts, self.available, alpha=alpha
            )
        for v in ts_vs:
            policies[f"TS (ʋ={v})"] = ThompsonSampling(
                self.device, self.contexts, self.available, v=v
            )
        
        # Simulate rounds
        rewards = np.empty((len(policies), rounds), dtype=bool)
        for round in trange(rounds, desc=f"Simulation {simulation_num}"):
            # Retrieve the random user for this round
            user_id = stream[round]

            # Simulate policies & record rewards
            for i, policy in enumerate(policies.values()):
                item_rec = policy.recommend(user_id)
                reward = self.rewards[user_id, item_rec]

                rewards[i, round] = reward
                policy.update(user_id, item_rec, reward)
        
        return rewards

    def run(
            self,
            seed_count: int = 10,
            rounds: int = 10000
    ):
        # Generate the random streams of users
        rng = np.random.default_rng()
        streams = rng.integers(low=0, high=SMALL_USERS, size=(seed_count, rounds))

        # Bandit policies
        e_greedy_epsilons = [0.01, 0.05, 0.1, 0.2]
        linucb_alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        ts_vs = [0.25, 0.5, 1.0, 2.0, 5.0]

        # Greedy & random + all bandit policies
        labels = (
            ["Greedy", "Random"]
            + [f"ε-greedy (ε={e})" for e in e_greedy_epsilons]
            + [f"LinUCB (α={a})" for a in linucb_alphas]
            + [f"TS (ʋ={v})" for v in ts_vs]
        )
        n_policies = len(labels)

        # Run simulation across seed_count many seeds
        all_rewards = np.empty((seed_count, n_policies, rounds), dtype=float)
        for seed_num, stream in enumerate(streams):
            all_rewards[seed_num] = self._run_one_seed(
                seed_num, stream, e_greedy_epsilons, linucb_alphas, ts_vs, rounds
            )
        
        # Now calculate means & standard deviations
        mean_rewards = all_rewards.mean(axis=0)
        std_errs =  all_rewards.std(axis=0) / np.sqrt(seed_count)

        # Plot results
        self._visualise(seed_count, labels, mean_rewards)