import torch
import torch.nn.functional as F
from torch import nn
import deep_linear_bandits.data as dlb_data
from deep_linear_bandits.data import NUM_USERS, NUM_ITEMS
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy

class UserTower(nn.Module):
    def __init__(
        self,
        cat_input_sizes,
        cat_emb_sizes,
        num_numeric_features,
        
        skip_towers: bool = False,

        id_emb_dims: int = 32,

        use_side_features: bool = True,
        hidden_sizes: list[int] = [128],
        output_size: int = 64,

        use_relu: bool = True,
        dropout: float = 0.2,
        use_l2_norm: bool = True
    ):
        super().__init__() # Needed for registering the nn.Modules correctly

        self.skip_towers = skip_towers

        # User ID embedding
        self.user_id_emb = nn.Embedding(
            NUM_USERS, id_emb_dims
        )

        self.use_side_features = use_side_features
        self.use_l2_norm = use_l2_norm

        if self.use_side_features:
            # Separate embedding network per categorical user feature
            self.cat_embs = nn.ModuleList([
                # Each nn.Embedding must take in the number of categories as input size,
                # then learns an output embedding of size e_size predetermined as 
                # appropriately wide for this categorical feature
                nn.Embedding(
                    i_size,
                    e_size
                ) for i_size, e_size in zip(
                    cat_input_sizes,
                    cat_emb_sizes
                )
            ])

            # Determine how wide the tower needs to be for the combined user input
            input_dim = (
                id_emb_dims # Width of embedded user ID vector
                + sum(cat_emb_sizes) # Width of embedded categorical features
                + num_numeric_features # Width of numeric features (directly fed to tower)
            )
        else:
            # Tower just needs to accept embedded user ID
            input_dim = id_emb_dims

        # Construct modules for the tower
        modules = []
        curr_input_dim = input_dim
        for i, hsize in enumerate(hidden_sizes):
            # Chains INPUT -> HIDDEN_1 -> ... -> HIDDEN_N
            modules.append(
                nn.Linear(curr_input_dim, hsize)
            )

            # If using ReLUs to learn nonlinearity, add them
            if use_relu:
                modules.append(
                    nn.ReLU()
                )
            
            # If using Dropouts to avoid overfitting, add them
            if dropout > 0.0:
                modules.append(
                    nn.Dropout(dropout)
                )

            # Ensures that the output size of the current layer aligns with the input size of the next
            curr_input_dim = hsize
        modules.append(
            # Add final link -> OUTPUT_SIZE
            nn.Linear(curr_input_dim, output_size)
        )

        # Now create the tower
        self.tower = nn.Sequential(*modules)
    
    def forward(
        self, 
        user_ids, 
        user_cat_feats=None, 
        user_numeric_feats=None
    ):
        # Embed the user ID into a dense vector
        id_emb = self.user_id_emb(user_ids)

        if self.use_side_features:
            # Embed the user's categorical features into dense vectors
            cat_embs = torch.cat([
                emb(user_cat_feats[:, i])
                for i, emb in enumerate(self.cat_embs)
            ], dim=1)

            # Combine user ID & features as total feature input for the User Tower
            tower_input = torch.cat(
                (id_emb, cat_embs, user_numeric_feats), dim=1
            )
        else:
            tower_input = id_emb

        # Pass user features into the User Tower
        if not self.skip_towers:
            user_embedding = self.tower(tower_input)
        else:
            user_embedding = tower_input

        # Optionally L2-normalise to coerce the model into learning meaningful relationships via cosine similarity
        # (otherwise it can just cheat in training by increasing/decreasing vector magnitudes)
        if self.use_l2_norm:
            user_embedding = F.normalize(user_embedding)

        return user_embedding # The learned user embedding vector

class ItemTower(nn.Module):
    def __init__(
        self,
        num_item_categories,

        skip_towers: bool = False,

        id_emb_dims: int = 32,
        item_cat_emb_dims: int = 16,

        use_side_features: bool = True,
        hidden_sizes: list[int] = [128],
        output_size: int = 64,

        use_relu: bool = True,
        dropout: float = 0.2,
        use_l2_norm: bool = True
    ):
        super().__init__()

        self.skip_towers = skip_towers
        self.use_side_features = use_side_features
        self.use_l2_norm = use_l2_norm

        # Item ID embedding
        self.item_id_emb = nn.Embedding(
            NUM_ITEMS, id_emb_dims
        )

        if self.use_side_features:
            # Embedding the sparse multi-hot categories to a dense representation
            # This is already a (num_item_categories)-long vector so use nn.Linear directly
            self.cat_emb = nn.Linear(
                num_item_categories, item_cat_emb_dims
            )

            # Determine how wide the tower's input size needs to be
            input_dim = id_emb_dims + item_cat_emb_dims
        else:
            input_dim = id_emb_dims

        # Construct modules for the tower
        modules = []
        curr_input_dim = input_dim
        for hsize in hidden_sizes:
            modules.append(
                nn.Linear(curr_input_dim, hsize)
            )

            if use_relu:
                modules.append(
                    nn.ReLU()
                )

            if dropout > 0.0:
                modules.append(
                    nn.Dropout(dropout)
                )

            curr_input_dim = hsize

        modules.append(
            nn.Linear(curr_input_dim, output_size)
        )

        # Now create the tower
        self.tower = nn.Sequential(*modules)
    
    def forward(
        self, 
        item_ids=None,
        item_categories=None
    ):
        # Embed the item ID into a dense vector
        id_emb = self.item_id_emb(item_ids)

        if self.use_side_features:
            # Embed the item's multi-hot categories into a single dense vector
            cat_emb = self.cat_emb(item_categories)

            # Combine item ID and categories into a combined item feature vector for the tower
            tower_input = torch.cat(
                (id_emb, cat_emb), dim=-1
            )
        else:
            tower_input = id_emb

        # Pass the item features through the tower to generate the final item embedding
        if not self.skip_towers:    
            item_embedding = self.tower(tower_input)
        else:
            item_embedding = tower_input

        # Optionally L2-normalise as done for the user embeddings too
        if self.use_l2_norm:
            item_embedding = F.normalize(item_embedding, dim=-1)

        return item_embedding # The learned item embedding vector

class TwoTower(nn.Module):
    def __init__(
        self,

        # User & item side features
        user_cat_input_sizes,
        user_cat_emb_sizes,
        user_num_numeric_features,
        num_item_categories,

        # Debug/eval option: disable towers entirely, just use nn.Embeddings all concatenated
        skip_towers: bool = False,

        # Sizes of intermediate representations
        id_emb_dims: int = 32,
        item_cat_emb_dims: int = 16,
        
        # Whether to use side features or not
        use_side_features: bool = True,

        # Hidden layer sizes & output size
        # Output default is 64 now; 128 or 256 are standard for massive user/item catalogues seen at e.g. Google
        hidden_sizes: list[int] = [128],
        output_size: int = 64,

        # Additional tower settings
        use_relu: bool = True,
        dropout: float = 0.2,
        use_l2_norm: bool = True,
        logit_temp: float = 0.07
    ):
        # Necessary for registering this module correctly
        super().__init__()

        self.logit_temp = logit_temp

        # Set up user tower
        self.user_tower = UserTower(
            cat_input_sizes=user_cat_input_sizes,
            cat_emb_sizes=user_cat_emb_sizes,
            num_numeric_features=user_num_numeric_features,

            skip_towers=skip_towers,

            id_emb_dims=id_emb_dims,

            use_side_features=use_side_features,

            hidden_sizes=hidden_sizes,
            output_size=output_size,

            use_relu=use_relu,
            dropout=dropout,
            use_l2_norm=use_l2_norm
        )

        # Set up item tower
        self.item_tower = ItemTower(
            num_item_categories=num_item_categories,

            skip_towers=skip_towers,

            id_emb_dims=id_emb_dims,
            item_cat_emb_dims=item_cat_emb_dims,

            use_side_features=use_side_features,
            
            hidden_sizes=hidden_sizes,
            output_size=output_size,

            use_relu=use_relu,
            dropout=dropout,
            use_l2_norm=use_l2_norm
        )
    
    def forward(
        self,

        user_ids,               # (B,)
        pos_item_ids,           # (B,)
        neg_item_ids,           # (K,)

        # Optional user side features (if in use)
        user_cat_feats=None,         # (B, num_user_cat_feats)
        user_numeric_feats=None,     # (B, num_numeric_feats)

        # Optional item side features (if in use)
        pos_item_categories=None,    # (B, num_item_categories)
        neg_item_categories=None    # (K, num_item_categories)
    ):
        # Generate user embedding
        user_emb = self.user_tower(
            user_ids,
            user_cat_feats,
            user_numeric_feats
        )  # (B, D)

        # Generate item embedding for the positive example
        pos_emb = self.item_tower(
            pos_item_ids,
            pos_item_categories
        )  # (B, D)

        # Generate item embedding for the negative examples
        neg_emb = self.item_tower(
            neg_item_ids,
            neg_item_categories
        )  # (K, D)

        # Element-wise dot product for each user with their positive
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)  # (B, 1)

        # Each user vs all K shared negatives
        neg_scores = user_emb @ neg_emb.T  # (B, K)

        # Similarity scores between users & items (logits, since they go into softmax for cross-entropy loss)
        logits = torch.cat([pos_scores, neg_scores], dim=1)  # (B, 1+K)

        # Use logit temperature to make the model "sharper" (more confident) in small similarity differences, otherwise it struggles to train in cases with very similar items that are misclassified as it doesn't penalise negatives enough + amplify positives enough in altering the vectors (since based on the softmax probabilities, which end up being all very similar for a long time)
        return logits / self.logit_temp

@torch.no_grad()
def compute_val_metrics(
    model: TwoTower,
    device: torch.device,

    # Features needed to calculate all user embeddings
    user_ids_t: torch.Tensor,
    user_cat_feats_t: torch.Tensor,
    user_numeric_feats_t: torch.Tensor,
    
    # Features needed to calculate all item embeddings
    item_ids_t: torch.Tensor,
    item_categories_t: torch.Tensor,

    # Known positive interactions already trained on, to mask out for validation
    train_pos_user_ids: np.ndarray,
    train_pos_item_ids: np.ndarray,

    # User IDs strictly of users in the validation set
    val_unique_user_ids: np.ndarray,

    # Known positive interactions held out for the validation set
    val_ground_truth: torch.Tensor,

    # K values to compute Recall@K & NDCG@K for
    k_values: list[int]
) -> tuple[list[float], list[float]]:
    # Embed all users
    user_embeddings = model.user_tower(
        user_ids_t,
        user_cat_feats_t,
        user_numeric_feats_t
    )

    # Embed all items
    item_embeddings = model.item_tower(
        item_ids_t,
        item_categories_t
    )

    # Compute all user-item similarity scores via dot products
    scores = user_embeddings @ item_embeddings.T

    # Mask out seen positives from training - no value in recommending an item already seen by the user
    scores[train_pos_user_ids, train_pos_item_ids] = -torch.inf

    # Filter out users that aren't in the validation set (not enough interactions to split on; not used in the metrics)
    val_ground_truth = val_ground_truth[val_unique_user_ids]
    scores = scores[val_unique_user_ids]

    # Get num of validation positives for each user; will be >=1 due to val user filtering
    val_counts = val_ground_truth.sum(dim=1)

    # Retrieve (new, non-training) items for each user that two-tower has ranked in the top-K
    # Use the max K value seen; lower K values can get their topk via slicing
    topk_all = scores.topk(max(k_values), dim=1).indices

    # Precompute discounts for each of the top-K items, for computing DCG
    # DCG = (1 if item relevant else 0) / (log2 (i+1)) where i is the position that the item occupies
    discount = (1.0 / torch.log2(torch.arange(2, max(k_values) + 2, device=device, dtype=torch.float32)))

    # Calculate Recall@K & NDCG@K for all K
    recall_results = []
    ndcg_results = []
    for k in k_values:
        topk = topk_all[:, :k]

        # Relevance matrix: for the chosen top K, what is their binary ground truth?
        # i.e. 1 if a validation positive for the user, 0 otherwise
        relevances = torch.gather(
            # i.e. in the ground truth matrix, look at each row; gather the indices that topk chose
            input=val_ground_truth,
            dim=1,
            index=topk
        )

        # Recall@K is just: (relevant entries in the top k / total relevant entries) averaged over all users
        recall_results.append((relevances.sum(dim=1) / val_counts).mean().item())

        # DCG@K: sum the (relevance / discount) values for each user's recommended items
        dcg = (relevances * discount[:k]).sum(dim=1)

        # IDCG@K: ideal DCG if all relevant items are at the top
        max_hits = val_counts.clamp(max=k).long() # Max hits that each user can have (since might be more than k)
        cum_discount = discount[:k].cumsum(dim=0) # DCG value for 1 hit at the top, then 2, ... k hits at the top
        idcg = cum_discount[max_hits - 1] # Index the correct IDCG value for each user

        # NDCG@K is just DCG@K/IDCG@K for each user; IDCG can't be 0 as all users have >=1 positive
        ndcg_results.append((dcg / idcg).mean().item())

    return recall_results, ndcg_results

def _score_weighted_obj(
    model: TwoTower,
    user_ids: torch.Tensor,
    user_cat_feats: torch.Tensor,
    user_numeric_feats: torch.Tensor,
    pos_item_ids: torch.Tensor,
    item_ids_t: torch.Tensor,
    item_categories_gpu: torch.Tensor,
    train_pos_mask: torch.Tensor,    # (NUM_USERS, NUM_ITEMS) bool - all training positives
    num_negatives: int,
    score_sharpness: float,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute logits and targets for a batch under score-weighted negative sampling.

    Each user's K negatives are drawn proportional to softmax(scores * score_sharpness),
    with all their known training positives masked out before sampling.

    Returns:
        logits: (B, 1+K) - positive score followed by K negative scores, temperature-scaled ready for softmax cross-entropy loss
        target: (B,)     - zeroes, since the positive is always at index 0
    """
    user_embs = model.user_tower(user_ids, user_cat_feats, user_numeric_feats)   # (B, D)
    all_item_embs = model.item_tower(item_ids_t, item_categories_gpu)            # (NUM_ITEMS, D)

    # Sample negative item IDs weighted by their (score_sharpness-scaled) scores, masking out positives
    with torch.no_grad():
        scores = user_embs @ all_item_embs.T                                        # (B, NUM_ITEMS)
        
        # Note scaling has to happen before mask, otherwise -torch.inf * 0 = NaN
        # This puts NaN in the softmax and then torch.multinomial isn't happy
        scaled = scores * score_sharpness
        scaled[train_pos_mask[user_ids]] = -torch.inf

        # Sample relative to the softmax-created probability distribution
        weights = torch.softmax(scaled, dim=-1)                   # (B, NUM_ITEMS)
        neg_indices = torch.multinomial(weights, num_negatives, replacement=False)  # (B, K)

    # Properly get item embeddings now
    pos_item_emb  = all_item_embs[pos_item_ids]                                  # (B, D)
    neg_item_embs = all_item_embs[neg_indices]                                   # (B, K, D)

    pos_scores = (user_embs * pos_item_emb).sum(dim=-1, keepdim=True)           # (B, 1); just dotprod between user and pos item
    neg_scores = torch.bmm(neg_item_embs, user_embs.unsqueeze(-1)).squeeze(-1)  # (B, K, D) bmm (B, D, 1) -> (B, K, 1) -> (B, K)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / model.logit_temp      # (B, 1+K)
    target = torch.zeros(logits.size(0), dtype=torch.long, device=device)       # Positives are all at position 0 of each of the B rows

    return logits, target

def _watch_ratio_weighted_obj(
    model: TwoTower,
    user_ids: torch.Tensor,
    user_cat_feats: torch.Tensor,
    user_numeric_feats: torch.Tensor,
    pos_item_ids: torch.Tensor,
    item_ids_t: torch.Tensor,
    item_categories_gpu: torch.Tensor,
    wr_weight_matrix: torch.Tensor,    # (NUM_USERS, NUM_ITEMS) precomputed per-user sampling weights
    num_negatives: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute logits and targets for a batch under watch-ratio-tiered hard negative sampling.

    Negatives are sampled per-user from wr_weight_matrix, which is set up to properly sample
    from each band with the probabilities specified in the CLI.

    Returns
        logits: (B, 1+K)    positive score followed by K negative scores, temperature-scaled ready for softmax cross-entropy loss
        target: (B,)        zeroes, since positive is always at index 0
    """
    user_embs = model.user_tower(user_ids, user_cat_feats, user_numeric_feats)   # (B, D)
    all_item_embs = model.item_tower(item_ids_t, item_categories_gpu)            # (NUM_ITEMS, D)

    # Look up precomputed per-user weights and sample K negatives per user
    with torch.no_grad():
        weights = wr_weight_matrix[user_ids]                                        # (B, NUM_ITEMS)
        neg_indices = torch.multinomial(weights, num_negatives, replacement=False)  # (B, K)

    pos_item_emb  = all_item_embs[pos_item_ids]                                  # (B, D)
    neg_item_embs = all_item_embs[neg_indices]                                   # (B, K, D)

    pos_scores = (user_embs * pos_item_emb).sum(dim=-1, keepdim=True)           # (B, 1); just dotprod between user and pos item
    neg_scores = torch.bmm(neg_item_embs, user_embs.unsqueeze(-1)).squeeze(-1)  # (B, K, D) bmm (B, D, 1) -> (B, K, 1) -> (B, K)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / model.logit_temp      # (B, 1+K)
    
    target = torch.zeros(logits.size(0), dtype=torch.long, device=device) # Positives at pos 0 for all users

    return logits, target

def train_two_tower(
    model: TwoTower,
    device: torch.device,

    metric_ks: list[int],
    best_k: int,

    train_loader: DataLoader,
    val_loader: DataLoader,

    training_set: dlb_data.KRBig,
    validation_set: dlb_data.KRBig,

    item_categories: torch.Tensor,      # CPU tensor (NUM_ITEMS, num_categories)
    user_cat_feats,                     # numpy (NUM_USERS, num_cat_feats)
    user_numeric_feats,                 # numpy (NUM_USERS, num_numeric_feats)

    epochs: int,
    num_negatives: int,
    negative_sampling: str,             # 'uniform', 'in-batch', 'score-weighted', 'watch-ratio', or 'full-softmax'
    score_sharpness: float,             # sharpness for score-weighted sampling: closer to 0 = uniform, higher = harder negatives
    train_wr_weights: torch.Tensor,     # (NUM_USERS, NUM_ITEMS) precomputed band weights for training
    val_wr_weights: torch.Tensor,       # (NUM_USERS, NUM_ITEMS) precomputed band weights for validation
    optimiser: torch.optim.Optimizer    # Adam or AdamW
) -> tuple[dict, TwoTower]:
    """
    Train the two-tower model for the given number of epochs, tracking training & validation loss alongside corpus-wide Recall@K and NDCG@K on the validation set each epoch.

    Returns:
        - dict of per-epoch metric lists
            - ['train_loss']
            - ['val_loss']
            - ['recall@{k}'] for all metric_ks
            - ['ndcg@{k}'] for all metric_ks
        - the best state of the passed model (by Recall@k for k=best_k)

    Full softmax:
        - What other negative sampling methods are attempting to approximate
        - Computationally feasible to compute on KuaiRec, even if it takes longer
        - Uses all other items as part of the cross-entropy loss calculation for each positive interaction
        - (including other positives for the user; that's what makes the denominator valid)

    Score-weighted negative sampling:
        - For each batch, each user gets K negatives with probability proportional to
          softmax(scores * score_sharpness) with their positives masked out
        - As such score_sharpness=0 gives uniform sampling / higher score_sharpness is more concentrated
          towards the top K scored items
        - This makes the negatives 'harder' i.e. ones that the model is struggling to discriminate
    
    Watch-ratio negative sampling:
        - Rather than 'hardness' being a measure of how much the model is misclassifying the sample (which it may be misclassifying samples that are noisy / weakly informative e.g. watch_ratio=1.9),
          hardness is a concrete measure based on the actual watch_ratios from the user for the videos
        - A watch_ratio of 0.3 is a much more significant negative in comparison to e.g. 1.9 which is nearly a positive
        - As such we should sample those lower watch_ratio items significantly more often per user
        - Per-user per-item sampling weights are precomputed in train_wr_weights/val_wr_weights
    """

    # Track best validation Recall@(best_k) to decide when to save the model, + save the model by its weights as that's all you need to revert it
    best_recall = -1.0
    best_weights = deepcopy(model.state_dict())

    # Use CrossEntropyLoss as the training objective
    loss_fn = nn.CrossEntropyLoss()

    # The item categories should be on the GPU for quick negative item sampling (they take up
    # ~1MB, negligible); make a copy for this, as the original needs to stay on the CPU for
    # multithreaded DataLoader workers
    item_categories_gpu = item_categories.to(device)

    # Create tensor versions of the user & item features (on GPU) for computing all user & item embeddings
    # at the end of an epoch; this is used for the corpus-wide (not batch-wide) validation metrics
    user_ids_t = torch.arange(NUM_USERS, dtype=torch.long, device=device)
    user_cat_feats_t = torch.tensor(user_cat_feats, device=device)
    user_numeric_feats_t = torch.tensor(user_numeric_feats, device=device)
    item_ids_t = torch.arange(NUM_ITEMS, dtype=torch.long, device=device)

    # Generate a validation ground-truth matrix: for each user, have a Boolean for their validation positives
    # This is used for computing Recall@K; does not have non-val users filtered out yet so that training positives
    # still line up for being filtered out (since otherwise the rows can't be interpreted as user IDs)
    val_ground_truth = torch.zeros(NUM_USERS, NUM_ITEMS, dtype=torch.bool, device=device)
    val_ground_truth[
        validation_set.user_ids,
        validation_set.item_ids
    ] = True

    # (NUM_USERS, NUM_ITEMS) bool mask of all training positives - used by score-weighted sampling
    # to exclude every known positive for a user, not just the one currently in the batch
    train_pos_mask = torch.zeros(NUM_USERS, NUM_ITEMS, dtype=torch.bool, device=device)
    train_pos_mask[training_set.user_ids, training_set.item_ids] = True

    # Combined mask for validation loss: excludes both training and validation positives so that
    # val loss negatives are drawn purely from items with no known positive signal
    val_pos_mask = train_pos_mask.clone()
    val_pos_mask[validation_set.user_ids, validation_set.item_ids] = True

    # Train model for multiple epochs, tracking both training & validation loss
    # Additionally track Recall@K & NDCG@K as proper validation metrics
    metrics = defaultdict(list)
    metrics["negative_sampling"] = negative_sampling
    metrics["num_negatives"] = num_negatives
    metrics["score_sharpness"] = score_sharpness
    for epoch in tqdm(range(1, epochs + 1), desc="Train/val epoch"):
        # Switch model into training mode
        model.train()

        # Train model on all training batches in this epoch; track training loss
        train_loss = 0
        for batch in train_loader:
            optimiser.zero_grad()

            if negative_sampling == 'full-softmax':
                # Full softmax: each user's positive interaction also gets all other items as part of the cross-entropy calculation
                # (doesn't try to approximate the loss, since KuaiRec is small enough to compute this)

                # All users from the batch
                user_embs = model.user_tower(
                    batch["user_id"].to(device),
                    batch["user_cat_feats"].to(device),
                    batch["user_numeric_feats"].to(device)
                )  # (B, D)

                # All items
                item_embs = model.item_tower(
                    item_ids_t,
                    item_categories_gpu
                ) # (num_items, D)

                # Compute all scores between the users & all items
                # Also apply logit temp
                logits = (user_embs @ item_embs.T) / model.logit_temp # (B, D) @ (D, num_items) -> (B, num_items)

                # Now the positive for each user is the one specified from the batch, so use this for the cross-entropy target
                target = batch["item_id"].to(device) # (B,) indices of the positive item to train on per user

            elif negative_sampling == 'in-batch':
                # In-batch negative sampling exists as a comparison point, but uniform negatives are seen to perform much better on KuaiRec
                # TwoTower is built for uniform negatives; this simply does the work of TwoTower's forward pass manually if it was using in-batch negative sampling -> better than writing two side-by-side TwoTower implementations, esp. given that in-batch negative sampling is not used for actual high-quality results

                # In-batch negative sampling: each user's positive item acts as a negative for all other users
                user_embs = model.user_tower(
                    batch["user_id"].to(device),
                    batch["user_cat_feats"].to(device),
                    batch["user_numeric_feats"].to(device)
                )  # (B, D)
                pos_item_embs = model.item_tower(
                    batch["item_id"].to(device),
                    batch["item_categories"].to(device)
                )  # (B, D)

                # Full similarity matrix: each user scored against all items in the batch
                logits = (user_embs @ pos_item_embs.T) / model.logit_temp  # (B, B)

                # The positive for user i is item i (on the diagonal)
                target = torch.arange(logits.size(0), device=device)
            
            elif negative_sampling == 'score-weighted':
                logits, target = _score_weighted_obj(
                    model,
                    batch["user_id"].to(device),
                    batch["user_cat_feats"].to(device),
                    batch["user_numeric_feats"].to(device),
                    batch["item_id"].to(device),
                    item_ids_t, item_categories_gpu,
                    train_pos_mask, num_negatives,
                    score_sharpness, device
                )
            
            elif negative_sampling == 'watch-ratio':
                logits, target = _watch_ratio_weighted_obj(
                    model,
                    batch["user_id"].to(device),
                    batch["user_cat_feats"].to(device),
                    batch["user_numeric_feats"].to(device),
                    batch["item_id"].to(device),
                    item_ids_t, item_categories_gpu,
                    train_wr_weights, num_negatives, device
                )
            
            else:
                # Uniform negative sampling: sample K random items shared across the batch
                neg_item_ids = torch.randint(
                    0, NUM_ITEMS, (num_negatives,), device=device
                )
                neg_item_categories = item_categories_gpu[neg_item_ids]

                # Calculate similarities (logits) between users and their positive items + the K negatives
                logits = model(
                    user_ids=batch["user_id"].to(device),
                    pos_item_ids=batch["item_id"].to(device),
                    neg_item_ids=neg_item_ids,

                    user_cat_feats=batch["user_cat_feats"].to(device),
                    user_numeric_feats=batch["user_numeric_feats"].to(device),

                    pos_item_categories=batch["item_categories"].to(device),
                    neg_item_categories=neg_item_categories
                )

                # The correct (positive) item for each user has the first logit; i.e. the 0th index on each row
                target = torch.zeros(
                    logits.size(0), dtype=torch.long, device=device
                )

            loss = loss_fn(logits, target)

            # Propagate loss backward, use optimiser to update model weights
            loss.backward()
            optimiser.step()

            train_loss += loss.item()
        train_loss /= len(train_loader) # Track average per-batch training loss

        # Switch model into evaluation mode: crucial so that dropout is disabled
        model.eval()

        # Check average per-batch validation loss after this epoch
        val_loss = 0
        with torch.no_grad(): # Not training so don't compute gradients for backprop
            for batch in val_loader:
                if negative_sampling == 'full-softmax':
                    # Same approach as in training

                    # Get user embs for the positive interactions to train on + all items for the full softmax loss
                    user_embs = model.user_tower(
                        batch["user_id"].to(device),
                        batch["user_cat_feats"].to(device),
                        batch["user_numeric_feats"].to(device)
                    )  # (B, D)
                    item_embs = model.item_tower(
                        item_ids_t,
                        item_categories_gpu
                    ) # (num_items, D)

                    # Compute all logits
                    logits = (user_embs @ item_embs.T) / model.logit_temp # (B, num_items)

                    # Positive for each user is the one specified in the batch
                    target = batch["item_id"].to(device) # (B,) indices of the positive item to train on per user

                elif negative_sampling == 'in-batch':
                    # In-batch: validation positives act as negatives for each other
                    user_embs = model.user_tower(
                        batch["user_id"].to(device),
                        batch["user_cat_feats"].to(device),
                        batch["user_numeric_feats"].to(device)
                    )
                    pos_item_embs = model.item_tower(
                        batch["item_id"].to(device),
                        batch["item_categories"].to(device)
                    )
                    logits = (user_embs @ pos_item_embs.T) / model.logit_temp
                    target = torch.arange(logits.size(0), device=device)
                elif negative_sampling == 'score-weighted':
                    # Score-weighted: same sharpness as training, both train & val positives masked
                    logits, target = _score_weighted_obj(
                        model,
                        batch["user_id"].to(device),
                        batch["user_cat_feats"].to(device),
                        batch["user_numeric_feats"].to(device),
                        batch["item_id"].to(device),
                        item_ids_t, item_categories_gpu,
                        val_pos_mask, num_negatives,
                        score_sharpness, device
                    )
                elif negative_sampling == 'watch-ratio':
                    # Watch-ratio: use validation-specific weights; training positives are masked out
                    logits, target = _watch_ratio_weighted_obj(
                        model,
                        batch["user_id"].to(device),
                        batch["user_cat_feats"].to(device),
                        batch["user_numeric_feats"].to(device),
                        batch["item_id"].to(device),
                        item_ids_t, item_categories_gpu,
                        val_wr_weights, num_negatives, device
                    )
                else:
                    # Uniform: K random catalogue items, same as training (no masking)
                    neg_item_ids = torch.randint(
                        0, NUM_ITEMS, (num_negatives,), device=device
                    )
                    neg_item_categories = item_categories_gpu[neg_item_ids]
                    logits = model(
                        user_ids=batch["user_id"].to(device),
                        pos_item_ids=batch["item_id"].to(device),
                        neg_item_ids=neg_item_ids,
                        user_cat_feats=batch["user_cat_feats"].to(device),
                        user_numeric_feats=batch["user_numeric_feats"].to(device),
                        pos_item_categories=batch["item_categories"].to(device),
                        neg_item_categories=neg_item_categories
                    )
                    target = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                val_loss += loss_fn(logits, target).item()
            val_loss /= len(val_loader) # Track per-batch average loss

        # Evaluate Recall@K & NDCG@K performance on the validation set
        recall, ndcg = compute_val_metrics(
            model,
            device,

            user_ids_t,
            user_cat_feats_t,
            user_numeric_feats_t,
            item_ids_t,
            item_categories_gpu,

            training_set.user_ids,
            training_set.item_ids,

            validation_set.unique_user_ids,

            val_ground_truth,

            k_values=metric_ks
        )

        for i, k in enumerate(metric_ks):
            metrics[f"recall@{k}"].append(recall[i])

        for i, k in enumerate(metric_ks):
            metrics[f"ndcg@{k}"].append(ndcg[i])

        # If best Recall@(best_k) has improved, save the model
        rk = metrics[f"recall@{best_k}"][-1]
        if rk > best_recall:
            best_weights = deepcopy(model.state_dict())
            best_recall = rk
            metrics["best_epoch"] = epoch

        # Collate results for later visualisation
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)

    print(f"Training complete - best Recall@{best_k}: {best_recall}")

    # Load the best model weights back in
    model.load_state_dict(
        best_weights
    )

    return dict(metrics), model

def visualise(
        metrics: dict,
        save_path: str,
        k_values: list[int],

        training_set: dlb_data.KRBig,
        validation_set: dlb_data.KRBig
) -> None:
    """
    Generates plots of the trained two-tower model's metrics across all of its train/val epochs, and saves them to disk.
    """

    # Compute expected Recall@K under a purely random recommendation policy, averaged over
    # all validation users - this is useful when visualising the Recall@K to show that the
    # model is learning significantly
    #
    # 1. Count number of training set positives
    train_counts = np.bincount(
        training_set.user_ids, minlength=NUM_USERS
    )

    # 2. Mask out users that aren't in the validation set
    train_counts = train_counts[validation_set.unique_user_ids]

    # 3. Calculate purely random baseline (masking out seen training positives)
    recall_baselines = [
        # Expected number of hits for K random items is exactly (K / total number of available items)
        (k / (NUM_ITEMS - train_counts)).mean() # Mean of the per-user baselines
        for k in k_values
    ]

    epochs = range(1, len(metrics["train_loss"]) + 1)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"] # PyPlot default colour cycle

    fig, (ax_loss, ax_recall, ax_ndcg) = plt.subplots(1, 3, figsize=(24, 10)) # Set up superplot

    # Plot training vs. validation loss
    ax_loss.plot(epochs, metrics["train_loss"], label="Training Loss")
    ax_loss.plot(epochs, metrics["val_loss"], label="Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean per-Batch Cross-Entropy Loss")
    ax_loss.set_title(
        f"Training & Validation Loss per Epoch ({metrics['negative_sampling']} negatives"
        +   (
                f", sharpness={metrics["score_sharpness"]})" 
                if metrics['negative_sampling'] == 'score-weighted'
                else ")"
            )
    )
    ax_loss.legend(fontsize=10, loc='upper right')
    ax_loss.grid(True, alpha=0.3)

    # Plot Recall@K over epochs with the random policy baselines
    for i, k in enumerate(k_values):
        colour = colours[i % len(colours)]
        ax_recall.plot(epochs, metrics[f"recall@{k}"], color=colour, label=f"Model Recall@{k}")
        ax_recall.axhline(
            recall_baselines[i],
            color=colour,
            linestyle=":",
            alpha=0.7,
            label=f"Random Policy Recall@{k} ({recall_baselines[i]:.4f})"
        )
    ax_recall.set_xlabel("Epoch")
    ax_recall.set_ylabel("Mean User Recall@K")
    ax_recall.set_title("Recall@K per Epoch (Validation Set)")
    ax_recall.legend(fontsize=9, loc='lower right')
    ax_recall.grid(True, alpha=0.3)

    # Plot NDCG@K over epochs
    for i, k in enumerate(k_values):
        colour = colours[i % len(colours)]
        ax_ndcg.plot(epochs, metrics[f"ndcg@{k}"], color=colour, label=f"NDCG@{k}")
    ax_ndcg.set_xlabel("Epoch")
    ax_ndcg.set_ylabel("Mean User NDCG@K")
    ax_ndcg.set_title("NDCG@K per Epoch (Validation Set)")
    ax_ndcg.legend(fontsize=10, loc='lower right')
    ax_ndcg.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()