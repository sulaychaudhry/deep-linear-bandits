"""
Handles loading & preprocessing the KuaiRec dataset.
"""

import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

WATCH_THRESHOLD = 2.0 # Strong binary signal for like/dislike implicitly derived from watch_ratio >= 2.0
BIG_MATRIX_TRAIN_SPLIT = 0.8 # Use 80% of big matrix interations for training, 20% for validation
BIG_MATRIX_MIN_N_SPLIT = 5 # Minimum number of interactions a user must have to partake in train-val split 

def load_kuairec_big(big_matrix_path):
    # Read in the big matrix
    big_matrix = pd.read_csv(
        big_matrix_path,
        usecols=[
            'user_id',
            'video_id',
            'watch_ratio'
        ]
    )

    # Filter for positive user-item interactions: these will be used to train the two-tower
    # Additionally remove any duplicate user-video interactions (some do exist in the dataset)
    big_matrix = big_matrix[big_matrix.watch_ratio >= WATCH_THRESHOLD].drop(
        columns=['watch_ratio']).drop_duplicates()
    
    # For the 80-20 train-val split, it will be stratified on users to mitigate cold-start and per-user
    # data split bias; some users have very low user-video interactions though and so are separated out
    #
    # This simply counts the entries per user_id and broadcasts a count<5 mask back to the original matrix shape
    low_interaction_mask = big_matrix.groupby('user_id')['user_id'].transform('count') < BIG_MATRIX_MIN_N_SPLIT

    # Perform the 80-20 train-val split, holding out those low-interaction users for additional training data
    split_candidates = big_matrix[~low_interaction_mask]
    bm_train, bm_val = train_test_split(
        split_candidates,
        train_size=BIG_MATRIX_TRAIN_SPLIT,
        shuffle=True,
        stratify=split_candidates['user_id']
    )
    bm_train = pd.concat([bm_train, big_matrix[low_interaction_mask]])

    # Convert to PyTorch dataset format & return
    return KRDataset(bm_train), KRDataset(bm_val)

class KRDataset(Dataset):
    def __init__(self, interactions_table):
        self.user_ids = interactions_table['user_id'].to_numpy()
        self.item_ids = interactions_table['video_id'].to_numpy()
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx]
        }