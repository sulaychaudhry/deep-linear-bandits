import torch
import torch.nn.functional as F
from torch import nn
import deep_linear_bandits.data as dlb_data
from deep_linear_bandits.data import NUM_USERS, NUM_ITEMS
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# Hyperparameters for the two-tower model
HYPERPARAMS = {
    # ----------------------------------
    # TRAINING HYPERPARAMETERS
    "NUM_NEGATIVES": 256,
    "BATCH_SIZE": 1024,
    "EPOCHS": 50,

    # ----------------------------------
    # TOWER ARCHITECTURE HYPERPARAMETERS

    # Embedding dimensions are defaulted to around 32 due to KuaiRec's relatively small size
    "USER_ID_EMB_DIM": 32,
    "ITEM_ID_EMB_DIM": 32,

    # There's 31 item categories, and items have 4 categories maximally; an 8-wide embedding might also be appropriate
    "ITEM_CAT_EMB_DIM": 16,

    "TOWER_HIDDEN_SIZE": 128, # Single hidden fully-connected (nn.Linear) layer of size 128, with a ReLU after
    "LOGIT_TEMPERATURE": 0.07, # Needed to discriminate fine differences in similarity scores between L2-normalised embeddings
    "DROPOUT": 0.2, # Helps with overfitting

    "OUTPUT_DIM": 64 # 64-wide embedding preferred for now over 128 or 256 as KuaiRec is quite a small dataset comparatively
}

class UserTower(nn.Module):
    def __init__(
        self,
        cat_input_sizes,
        cat_emb_sizes,
        num_numeric_features
    ):
        super().__init__() # Needed for registering the nn.Modules correctly

        # User ID embedding
        self.user_id_emb = nn.Embedding(
            7176, HYPERPARAMS["USER_ID_EMB_DIM"]
        )

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
            HYPERPARAMS["USER_ID_EMB_DIM"] # Width of embedded user ID vector
            + sum(cat_emb_sizes) # Width of embedded categorical features
            + num_numeric_features # Width of numeric features (directly fed to tower)
        )

        # The user tower itself
        self.tower = nn.Sequential(
            nn.Linear(input_dim, HYPERPARAMS["TOWER_HIDDEN_SIZE"]),
            nn.ReLU(),
            nn.Dropout(HYPERPARAMS["DROPOUT"]),
            nn.Linear(HYPERPARAMS["TOWER_HIDDEN_SIZE"], HYPERPARAMS["OUTPUT_DIM"])
        )
    
    def forward(
        self, 
        user_ids, 
        user_cat_feats, 
        user_numeric_feats
    ):
        # Embed the user ID into a dense vector
        id_emb = self.user_id_emb(user_ids)

        # Embed the user's categorical features into dense vectors
        cat_embs = torch.cat([
            emb(user_cat_feats[:, i])
            for i, emb in enumerate(self.cat_embs)
        ], dim=1)

        # Combine user ID & categorical features as total feature input for the User Tower
        tower_input = torch.cat(
            (id_emb, cat_embs, user_numeric_feats), dim=1
        )

        # Pass user features into the User Tower & L2-normalise to coerce the model into learning meaningful relationships
        # (otherwise it can just cheat in training by increasing/decreasing vector magnitudes)
        user_embedding = F.normalize(
            self.tower(tower_input)
        )

        return user_embedding # The learned user embedding vector

class ItemTower(nn.Module):
    def __init__(
        self,
        num_item_categories
    ):
        super().__init__()

        # Item ID embedding
        self.item_id_emb = nn.Embedding(
            10728, HYPERPARAMS["ITEM_ID_EMB_DIM"]
        )

        # Embedding the sparse multi-hot categories to a dense representation
        # This is already a (num_item_categories)-long vector so use nn.Linear directly
        self.cat_emb = nn.Linear(
            num_item_categories, HYPERPARAMS["ITEM_CAT_EMB_DIM"]
        )

        # Determine how wide the tower's input size needs to be
        input_dim = HYPERPARAMS["ITEM_ID_EMB_DIM"] + HYPERPARAMS["ITEM_CAT_EMB_DIM"]

        # The item tower itself
        self.tower = nn.Sequential(
            nn.Linear(input_dim, HYPERPARAMS["TOWER_HIDDEN_SIZE"]),
            nn.ReLU(),
            nn.Dropout(HYPERPARAMS["DROPOUT"]),
            nn.Linear(HYPERPARAMS["TOWER_HIDDEN_SIZE"], HYPERPARAMS["OUTPUT_DIM"])
        )
    
    def forward(
        self, 
        item_ids,
        item_categories
    ):
        # Embed the item ID into a dense vector
        id_emb = self.item_id_emb(item_ids)

        # Embed the item's multi-hot categories into a single dense vector
        cat_emb = self.cat_emb(item_categories)

        # Combine item ID and categories into a combined item feature vector for the tower
        tower_input = torch.cat(
            (id_emb, cat_emb), dim=-1
        )

        # Pass the item features through the tower to generate the final item embedding
        # & L2-normalise as done for the user embeddings too
        item_embedding = F.normalize(
            self.tower(tower_input), dim=-1
        )

        return item_embedding # The learned item embedding vector

class TwoTower(nn.Module):
    def __init__(
        self,
        user_cat_input_sizes,
        user_cat_emb_sizes,
        user_num_numeric_features,
        num_item_categories
    ):
        super().__init__()

        self.user_tower = UserTower(
            user_cat_input_sizes,
            user_cat_emb_sizes,
            user_num_numeric_features
        )

        self.item_tower = ItemTower(
            num_item_categories
        )
    
    def forward(
        self,

        user_ids,               # (B,)
        user_cat_feats,         # (B, num_user_cat_feats)
        user_numeric_feats,     # (B, num_numeric_feats)

        pos_item_ids,           # (B,)
        pos_item_categories,    # (B, num_item_categories)

        neg_item_ids,           # (K,)
        neg_item_categories,    # (K, num_item_categories)
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

        return logits / HYPERPARAMS["LOGIT_TEMPERATURE"]

@torch.no_grad()
def recall_at_k(
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

    # Known positive interactions held out for the validation set, to base Recall@K off
    vaL_pos_user_ids: np.ndarray,
    val_pos_item_ids: np.ndarray,

    # K values to compute Recall@K for
    k: list[int] = [10, 50]
):
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

    # Generate a validation ground-truth matrix: for each user, have a Boolean for their validation positives
    val_ground_truth = torch.zeros(NUM_USERS, NUM_ITEMS, dtype=torch.bool, device=device)
    val_ground_truth[vaL_pos_user_ids, val_pos_item_ids] = True

    # Filter out users that aren't in the validation set (not enough interactions to split on; not used in Recall@K)
    val_ground_truth = val_ground_truth[val_unique_user_ids]
    scores = scores[val_unique_user_ids]

    # Get num of validation positives for each user; will be >=1 due to val user filtering
    val_counts = val_ground_truth.sum(dim=1)

    # Retrieve (new, non-training) items for each user that two-tower has ranked in the top-K
    # Use the max K value seen; lower K values can get their topk via slicing
    topk = scores.topk(max(k), dim=1).indices

    # Calculate Recall@K for all K
    results = []
    for ki in k:
        topki = topk[:, :ki]

        # Check how many of these are actually validation positives for each user
        hit_counts = torch.gather(
            # i.e. in the ground truth matrix, look at each row; gather only the indices that topk chose
            input=val_ground_truth,
            dim=1,
            index=topki
        ).sum(dim=1)

        results.append((hit_counts / val_counts).mean().item()) # Calculate recall hit percentage, store result

    return results

def generate_two_tower_model(
    device: torch.device             # Device to train the model on (GPU if available)
) -> TwoTower:                       # The trained two-tower model

    # Get training & validation (positive) user-item interactions from KuaiRec-Big
    pos_intrs_train, pos_intrs_val = dlb_data.preprocess_krbig_interactions()

    # Get user side features: the categorical and numeric user features
    (user_cat_feats, user_cat_sizes), user_numeric_feats = dlb_data.preprocess_user_features()

    # Get item side features: the item categories
    item_categories = dlb_data.preprocess_item_categories()

    # Create two-tower model & move to GPU
    model = TwoTower(
        user_cat_sizes,              # Sizes of each categorical user feature
        [8] * len(user_cat_sizes),   # Embedding widths for each categorical user feature -> TODO!!!!! / make it calc
        user_numeric_feats.shape[1], # Number of numeric user features
        item_categories.shape[1]     # Number of item categories
    ).to(device)

    # Compile model for significantly quicker forward & backward passes
    model.compile()

    # Pass training & validation datasets through KRBig for PyTorch batching compatibility
    training_set = dlb_data.KRBig(
        pos_intrs_train,
        user_cat_feats,
        user_numeric_feats,
        item_categories
    )
    validation_set = dlb_data.KRBig(
        pos_intrs_val,
        user_cat_feats,
        user_numeric_feats,
        item_categories
    )

    # Set up DataLoaders for dispatching training & validation batches
    # Use multithreading & pinned memory (between RAM & CUDA) for much quicker retrieval
    train_loader = DataLoader(
        training_set,
        batch_size=HYPERPARAMS["BATCH_SIZE"],
        shuffle=True, # Shuffle per epoch to reduce it from fitting on data order
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=HYPERPARAMS["BATCH_SIZE"],
        shuffle=False, # Don't shuffle per-epoch for validation, not necessary & has been shuffled during data split
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Use CrossEntropyLoss as expected; use Adam with defaults to optimise the model
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    # The item categories should be on the GPU for quick negative item sampling (they take up
    # ~1MB, negligible); make a copy for this, as the original needs to stay on the CPU for
    # mulithreaded DataLoader workers
    item_categories_gpu = item_categories.to(device)

    # Create tensor versions of the user & item features (on GPU) for computing all user & item embeddings
    # at the end of an epoch; this is used for the corpus-wide (not batch-wide) validation metrics
    user_ids_t = torch.arange(NUM_USERS, dtype=torch.long, device=device)
    user_cat_feats_t = torch.tensor(user_cat_feats, device=device)
    user_numeric_feats_t = torch.tensor(user_numeric_feats, device=device)
    item_ids_t = torch.arange(NUM_ITEMS, dtype=torch.long, device=device)

    # Train model for multiple epochs, tracking both training & validation loss
    # Additionally track Recall@K as a proper validation metric
    metrics = defaultdict(list)
    for epoch in range(1, HYPERPARAMS["EPOCHS"] + 1):
        # Switch model into training mode
        model.train()

        # Train model on all training batches in this epoch; track training loss
        train_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{HYPERPARAMS["EPOCHS"]} (train)"
        ):
            opt.zero_grad() # Zero the optimiser gradients

            # The batch contains positive user-item interactions sequentially, but doesn't give us negatives
            # These are sampled uniformly; in-batch negative sampling is seen to perform really poorly on KuaiRec
            neg_item_ids = torch.randint(
                0, NUM_ITEMS, (HYPERPARAMS["NUM_NEGATIVES"],), device=device
            )
            neg_item_categories = item_categories_gpu[neg_item_ids]
            
            # Calculate similarities (logits) between users and their positive items + the K negatives
            logits = model(
                # User features from the batch
                batch["user_id"].to(device),
                batch["user_cat_feats"].to(device),
                batch["user_numeric_feats"].to(device),

                # Positive item features from the batch
                batch["item_id"].to(device),
                batch["item_categories"].to(device),

                # Uniformly sampled negative items
                neg_item_ids,
                neg_item_categories
            )

            # The correct (positive) item for each user has the first logit; i.e. the 0th index on each row
            target = torch.zeros(
                logits.size(0), dtype=torch.long, device=device
            )
            loss = loss_fn(logits, target)

            # Propagate loss backward, use optimiser to update model weights
            loss.backward()
            opt.step()

            train_loss += loss.item()
        train_loss /= len(train_loader) # Track average per-batch training loss

        # Switch model into evaluation mode: crucial so that dropout is disabled
        model.eval()

        # Check average per-batch validation loss after this epoch
        val_loss = 0
        with torch.no_grad(): # Not training so don't compute gradients for backprop
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch}/{HYPERPARAMS["EPOCHS"]} (val)"
            ):
                # Again, uniformly sample negatives (positive objective is strictly from validation set though now)
                neg_item_ids = torch.randint(
                    0, NUM_ITEMS, (HYPERPARAMS["NUM_NEGATIVES"],), device=device
                )
                neg_item_categories = item_categories_gpu[neg_item_ids]

                # Compute similarities between user embedding & positive item embedding + uniform negatives
                logits = model(
                    # User features from the batch
                    batch["user_id"].to(device),
                    batch["user_cat_feats"].to(device),
                    batch["user_numeric_feats"].to(device),

                    # Positive item features from the batch
                    batch["item_id"].to(device),
                    batch["item_categories"].to(device),

                    # Uniformly sampled negative items
                    neg_item_ids,
                    neg_item_categories
                )

                # Same target as in training: the first item is the positive one
                target = torch.zeros(
                    logits.size(0), dtype=torch.long, device=device
                )
                
                val_loss += loss_fn(logits, target).item()
            val_loss /= len(val_loader) # Track per-batch average loss

        print(f"Epoch {epoch} average batch loss (train): {train_loss}")
        print(f"Epoch {epoch} average batch loss (val): {val_loss}")

        # Evaluate recall@K performance on the validation set
        recall = recall_at_k(
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

            validation_set.user_ids,
            validation_set.item_ids
        )

        print(f"Epoch {epoch} Recall@10 (val, avg. per-user): {recall[0]}")
        print(f"Epoch {epoch} Recall@50 (val, avg. per-user): {recall[1]}")

        # Collate results for later visualisation
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["recall@10"].append(recall[0])
        metrics["recall@50"].append(recall[1])
    
    # Visualise two-tower training & validation metrics
