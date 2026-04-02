import numpy as np
import torch
from deep_linear_bandits.data import KRSmall
from tqdm import trange
import matplotlib.pyplot as plt
from datetime import datetime

SMALL_USERS = 1411
SMALL_ITEMS = 3327

def gini_coefficient(recommendations: np.ndarray) -> float:
    """
    Computes the Gini coefficient over recommendation frequencies.
    A value of 0 means perfectly uniform coverage; 1 means all recommendations go to a single item.
    """

    # Bin the item ID counts & collect sum
    counts = np.bincount(recommendations, minlength=SMALL_ITEMS).astype(np.float64)
    total = counts.sum()
    if total == 0: return 0.0

    # Use this efficient formula: https://en.wikipedia.org/wiki/Gini_coefficient#Alternative_expressions
    y = np.sort(counts)
    n = len(y)
    index = np.arange(1, n + 1)
    return ((2 * np.dot(index, y)) / (n * total) - (n + 1) / n)


def longtail_coverage(
    recommendations: np.ndarray,
    item_popularities: np.ndarray,
    percentile: float = 80.0
) -> float:
    """
    Fraction of long-tail items i.e. items whose popularity is strictly below `percentile` that were recommended at least once.
    """

    # Determine exactly which items fall below the threshold
    threshold = np.percentile(item_popularities, percentile)
    longtail_mask = item_popularities < threshold
    longtail_items = np.where(longtail_mask)[0]
    if len(longtail_items) == 0: return 0.0

    # Check which ones were recommended & count
    recommended_set = np.unique(recommendations)
    covered = np.isin(longtail_items, recommended_set).sum()
    return covered / len(longtail_items)


def average_recommendation_popularity(
    recommendations: np.ndarray,
    item_popularities: np.ndarray
) -> float:
    """
    Average popularity of recommended items across all rounds.
    """
    return item_popularities[recommendations].mean()


def compute_all_metrics(
    all_rewards: np.ndarray,           # (seeds, policies, rounds)
    all_regrets: np.ndarray,           # (seeds, policies, rounds)
    all_recommendations: np.ndarray,   # (seeds, policies, rounds)
    item_popularities: np.ndarray,     # (n_items,)
    longtail_percentile: float = 80.0
) -> dict:
    """
    Aggregates all metrics (rewards, regrets, diversity) across seeds.

    Returns a dict with mean/std for cumulative rewards/regrets (full curves + final values) and mean/std
    for Gini, long-tail coverage, and ARP per policy.
    """
    seed_count, n_policies, _ = all_rewards.shape

    all_cum_rewards = np.cumsum(all_rewards, axis=2)
    all_cum_regrets = np.cumsum(all_regrets, axis=2)

    # Per-seed-per-policy diversity metrics
    all_gini = np.empty((seed_count, n_policies))
    all_coverage = np.empty((seed_count, n_policies))
    all_arp = np.empty((seed_count, n_policies))
    for s in range(seed_count):
        for p in range(n_policies):
            recs = all_recommendations[s, p]
            all_gini[s, p] = gini_coefficient(recs)
            all_coverage[s, p] = longtail_coverage(recs, item_popularities, longtail_percentile)
            all_arp[s, p] = average_recommendation_popularity(recs, item_popularities)

    return {
        "mean_cumulative_rewards":        all_cum_rewards.mean(axis=0).tolist(),
        "std_cumulative_rewards":         all_cum_rewards.std(axis=0).tolist(),

        "mean_cumulative_regrets":        all_cum_regrets.mean(axis=0).tolist(),
        "std_cumulative_regrets":         all_cum_regrets.std(axis=0).tolist(),

        "mean_final_cumulative_rewards":  all_cum_rewards.mean(axis=0)[:, -1].tolist(),
        "std_final_cumulative_rewards":   all_cum_rewards.std(axis=0)[:, -1].tolist(),

        "mean_final_cumulative_regrets":  all_cum_regrets.mean(axis=0)[:, -1].tolist(),
        "std_final_cumulative_regrets":   all_cum_regrets.std(axis=0)[:, -1].tolist(),

        "mean_gini":                      all_gini.mean(axis=0).tolist(),
        "std_gini":                       all_gini.std(axis=0).tolist(),

        "mean_longtail_coverage":         all_coverage.mean(axis=0).tolist(),
        "std_longtail_coverage":          all_coverage.std(axis=0).tolist(),

        "mean_arp":                       all_arp.mean(axis=0).tolist(),
        "std_arp":                        all_arp.std(axis=0).tolist(),
    }

def build_two_tower_contexts(
        user_embeddings: torch.Tensor,  # (U, D)
        item_embeddings: torch.Tensor,  # (I, D)
        include_product: bool = True
) -> torch.Tensor:
    """
    Build context vectors from two-tower embeddings.

    Constructs (U, I, D_ctx) context tensors by concatenating:
        1. user embedding repeated across items    (U, I, D)
        2. item embedding repeated across users    (U, I, D)

        (if include_product=True)
        3. element-wise product i.e. Hadamard      (U, I, D)

    A given user-item context vector ends with size D_ctx = 2*D if no Hadamard, else D_ctx = 3*D.
    """

    # Precompute all context vectors across all users and all items; this is large (over 3GB) but will fit in VRAM
    u = user_embeddings.unsqueeze(1)  # (U, 1, D)
    i = item_embeddings.unsqueeze(0)  # (1, I, D)
    parts = [
        u.expand(-1, item_embeddings.shape[0], -1),  # (U, I, D)
        i.expand(user_embeddings.shape[0], -1, -1),  # (U, I, D)
    ]
    if include_product:
        parts.append(u * i)  # (U, I, D) via broadcasting
    return torch.cat(parts, dim=-1)  # (U, I, D_ctx)

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
        self.available = available.detach().clone().cpu().numpy()
        self.rng = rng

    def recommend(self, user_id:int) -> int:
        available = self.available[user_id]

        # Pick a random available item, using flatnonzero to efficiently get item IDs of available items
        available = np.flatnonzero(available)
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
        except torch.linalg.LinAlgError:
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
            contexts: torch.Tensor,
            user_embeddings: torch.Tensor,
            item_embeddings: torch.Tensor
    ):
        self.device = device
        self.contexts = contexts

        # Compute a matrix to indicate all interactions that are 'valid' i.e. don't need masking out
        # This is because despite 99.7% density, some interactions are missing
        self.available = torch.zeros(SMALL_USERS, SMALL_ITEMS, dtype=torch.bool, device=device)
        self.available[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = True

        # Also compute a ground truth reward matrix for all positive user-item interactions
        self.rewards = np.zeros((SMALL_USERS, SMALL_ITEMS), dtype=bool)
        self.rewards[small_matrix.intr_new_uids, small_matrix.intr_new_iids] = small_matrix.intr_signals

        # Precompute dot-product similarity scores for GreedyPolicy
        # This can be used immediately to derive the item IDs for GreedyPolicy as it has fixed behaviour
        dot_products = user_embeddings @ item_embeddings.T
        self.greedy_items = torch.sort(dot_products, dim=1, descending=True).indices.cpu().numpy()

    @staticmethod
    def _visualise_rewards(
            seed_count: int,
            labels: list[str],
            mean_rewards: np.ndarray[float],
            output_dir: str
    ):
        # Get cumulative rewards (from the means)
        cum_reward_means = np.cumsum(mean_rewards, axis=1)
        rounds = np.arange(1, mean_rewards.shape[1] + 1)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        if seed_count > 1:
            fig.suptitle(f"Policy simulation: cumulative bandit reward (averaged over {seed_count} simulations)")
        else:
            fig.suptitle(f"Policy simulation: cumulative bandit reward (shared random seed)")

        # Set up all three subplots with Greedy & Random baselines
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
            ax1.plot(rounds, cum_reward_means[i], color=colours[j % len(colours)], label=label)
            ax1.set_title("ε-greedy rewards")
            ax1.legend()
            j += 1

        # Plot LinUCB rewards
        j, colours = 0, ['#fddbb4', '#f5a962', '#E8612C', '#a83a10', '#5c1a04']
        for i, label in enumerate(labels):
            if not label.startswith("LinUCB"): continue
            ax2.plot(rounds, cum_reward_means[i], color=colours[j % len(colours)], label=label)
            ax2.set_title("LinUCB rewards")
            ax2.legend()
            j += 1

        # Plot Thompson Sampling rewards
        j, colours = 0, ['#a8d8a8', '#6AAB6A', '#3d7a3d', '#1e4f1e', '#0a2a0a']
        for i, label in enumerate(labels):
            if not label.startswith("TS"): continue
            ax3.plot(rounds, cum_reward_means[i], color=colours[j % len(colours)], label=label)
            ax3.set_title("Thompson Sampling rewards")
            ax3.legend()
            j += 1

        plt.tight_layout()
        plt.savefig(output_dir + 'rewards.png')
        plt.close()

    @staticmethod
    def _visualise_regrets(
            seed_count: int,
            labels: list[str],
            mean_regrets: np.ndarray,
            output_dir: str
    ):
        # Get cumulative regrets (from the means)
        cum_regret_means = np.cumsum(mean_regrets, axis=1)
        rounds = np.arange(1, mean_regrets.shape[1] + 1)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        if seed_count > 1:
            fig.suptitle(f"Policy simulation: cumulative bandit regret (averaged over {seed_count} simulations)")
        else:
            fig.suptitle(f"Policy simulation: cumulative bandit regret (shared random seed)")

        # Set up all three subplots with Random & Greedy baselines
        for ax in (ax1, ax2, ax3):
            ax.plot(rounds, cum_regret_means[1], color='#aaaaaa', label="Random", linestyle='--')
            ax.plot(rounds, cum_regret_means[0], color='#444444', label="Greedy", linestyle='--')
            ax.set_xlabel("Round")
            ax.set_ylabel("Cumulative regret")
            ax.grid(True, alpha=0.3)

        # Plot ε-greedy regrets
        j, colours = 0, ['#c6d9f7', '#7aaede', '#4878CF', '#1e3f7a']
        for i, label in enumerate(labels):
            if not label.startswith("ε-greedy"): continue
            ax1.plot(rounds, cum_regret_means[i], color=colours[j % len(colours)], label=label)
            ax1.set_title("ε-greedy regret")
            ax1.legend()
            j += 1

        # Plot LinUCB regrets
        j, colours = 0, ['#fddbb4', '#f5a962', '#E8612C', '#a83a10', '#5c1a04']
        for i, label in enumerate(labels):
            if not label.startswith("LinUCB"): continue
            ax2.plot(rounds, cum_regret_means[i], color=colours[j % len(colours)], label=label)
            ax2.set_title("LinUCB regret")
            ax2.legend()
            j += 1

        # Plot Thompson Sampling regrets
        j, colours = 0, ['#a8d8a8', '#6AAB6A', '#3d7a3d', '#1e4f1e', '#0a2a0a']
        for i, label in enumerate(labels):
            if not label.startswith("TS"): continue
            ax3.plot(rounds, cum_regret_means[i], color=colours[j % len(colours)], label=label)
            ax3.set_title("Thompson Sampling regret")
            ax3.legend()
            j += 1

        plt.tight_layout()
        plt.savefig(output_dir + 'regrets.png')
        plt.close()

    def _run_one_seed(
            self,
            simulation_num: int,
            stream: np.ndarray,
            e_greedy_epsilons: list[float],
            linucb_alphas: list[float],
            ts_vs: list[float],
            lam: float,
            rounds: int,
            seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Derive independent per-policy RNGs from the seed so policies don't share state; more methodologically sound
        rng = np.random.default_rng(seed)
        n_rngs = 1 + len(e_greedy_epsilons)  # RandomPolicy + each EpsilonGreedy
        policy_rngs = rng.spawn(n_rngs)

        # Set up policies
        policies = {
            "Greedy": GreedyPolicy(self.greedy_items, self.available.cpu().numpy()),
            "Random": RandomPolicy(self.available, policy_rngs[0])
        }
        for i, eps in enumerate(e_greedy_epsilons):
            policies[f"ε-greedy (ε={eps})"] = EpsilonGreedy(
                self.device, self.contexts, self.available, policy_rngs[1 + i], epsilon=eps, lam=lam
            )
        for alpha in linucb_alphas:
            policies[f"LinUCB (α={alpha})"] = LinUCB(
                self.device, self.contexts, self.available, alpha=alpha, lam=lam
            )
        for v in ts_vs:
            policies[f"TS (ʋ={v})"] = ThompsonSampling(
                self.device, self.contexts, self.available, v=v, lam=lam
            )

        # Calculate the oracle: how many positives exist per user?
        pos_count = self.rewards.sum(axis=1)
        pos_count = np.tile(pos_count, (len(policies), 1)) # Separate oracle copy for each policy to update

        # Simulate rounds
        rewards = np.empty((len(policies), rounds), dtype=bool)
        regrets = np.empty((len(policies), rounds), dtype=bool)
        recommendations = np.empty((len(policies), rounds), dtype=np.int16)
        for round in trange(rounds, desc=f"Simulation {simulation_num}"):
            # Retrieve the random user for this round
            user_id = stream[round]

            # Simulate policies & record rewards
            for i, policy in enumerate(policies.values()):
                # Check oracle for this policy: is a reward achievable?
                oracle_reward = pos_count[i, user_id] > 0

                # Policy chooses item, observe its reward
                item_rec = policy.recommend(user_id)
                reward = self.rewards[user_id, item_rec]

                # Record reward, regret, and recommendation
                rewards[i, round] = reward
                regrets[i, round] = oracle_reward and not reward # Reward available but not achieved
                recommendations[i, round] = item_rec

                # Update oracle
                if reward: pos_count[i, user_id] -= 1

                # Update policy (allow bandits to update their ridge regression)
                policy.update(user_id, item_rec, reward)

        return rewards, regrets, recommendations

    def run(
            self,
            seed_count: int = 1,
            rounds: int = 10000,
            e_greedy_epsilons: list[float] = [0.01, 0.05, 0.1, 0.2],
            linucb_alphas: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
            ts_vs: list[float] = [0.25, 0.5, 1.0, 2.0, 5.0],
            lam: float = 1.0,
            seed: int | None = None,      # if None, OS entropy will be used instead
            seed_index: int | None = None  # if set, run only this seed (for Slurm); otherwise run all seeds
    ) -> dict:
        # Derive deterministic per-seed RNG seeds & user streams from the master seed
        # This ensures that splitting across Slurm jobs with --seed-index produces the same results as a single run
        rng = np.random.default_rng(seed)
        child_seeds = rng.integers(0, 2**32, size=seed_count).tolist()
        streams = rng.integers(low=0, high=SMALL_USERS, size=(seed_count, rounds))

        # Policy labels
        labels = (
            ["Greedy", "Random"]
            + [f"ε-greedy (ε={e})" for e in e_greedy_epsilons]
            + [f"LinUCB (α={a})" for a in linucb_alphas]
            + [f"TS (ʋ={v})" for v in ts_vs]
        )

        # Dispatched over multiple Slurm jobs - this is just one of them
        if seed_index is not None:
            idx = seed_index if seed_index is not None else 0
            rewards, regrets, recommendations = self._run_one_seed(
                idx, streams[idx], e_greedy_epsilons, linucb_alphas, ts_vs, lam, rounds, child_seeds[idx]
            )
            return {"labels": labels, "rewards": rewards, "regrets": regrets, "recommendations": recommendations}

        # All seeds are within this process
        else:
            n_policies = len(labels)
            all_rewards = np.empty((seed_count, n_policies, rounds), dtype=float)
            all_regrets = np.empty((seed_count, n_policies, rounds), dtype=float)
            all_recommendations = np.empty((seed_count, n_policies, rounds), dtype=np.int16)
            for i in range(seed_count):
                all_rewards[i], all_regrets[i], all_recommendations[i] = self._run_one_seed(
                    i, streams[i], e_greedy_epsilons, linucb_alphas, ts_vs, lam, rounds, child_seeds[i]
                )

            return {
                "labels": labels,
                "mean_rewards": all_rewards.mean(axis=0),
                "mean_regrets": all_regrets.mean(axis=0),
                "all_rewards": all_rewards,
                "all_regrets": all_regrets,
                "all_recommendations": all_recommendations,
            }