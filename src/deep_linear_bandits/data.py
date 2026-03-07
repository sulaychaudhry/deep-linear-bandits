import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader

NUM_USERS = 7176
NUM_ITEMS = 10728

DATA_DIR = "/home/sulay/deep-linear-bandits/kuairec/data/"

def preprocess_krbig_interactions():
    """
    Preprocess KuaiRec-Big user-item interactions into the training &
    validation set; only "strongly positive" interactions (of which there
    are about 800k) are kept: those with watch_ratio >= 2.0 as suggested in
    the KuaiRec paper.

    The training set has size 80%, leaving a validation set of size 20%.
    These splits are per-user to avoid bias against cold users due to
    random data split; not temporal as KuaiRec's small matrix can only
    be used for offline evaluation if interactions are treated
    independently of their timestamps - future work can focus on temporal
    factors.

    Returns
    bm_train: Pandas dataframe containing training interactions
    bm_val: Pandas dataframe containing validation interactions

    Each of these dataframes have cols
        | user_id | video_id |
    """

    # Read in KuaiRec-Big
    bm = pd.read_csv(
        DATA_DIR + "big_matrix.csv",
        usecols=["user_id", "video_id", "watch_ratio"]
    )

    # Filter for the "strongly positive" signal: watch_ratio >= 2.0
    # Also removes duplicate user-item interactions
    bm = (
        bm[bm["watch_ratio"] >= 2.0]
        .drop(columns=["watch_ratio"])
        .drop_duplicates()
    )

    # Users with less than 5 interactions (after signal filtering)
    # contribute purely to training; they only constitute a small
    # portion of the dataset.
    user_counts = bm["user_id"].value_counts()
    low_users = user_counts[user_counts < 5].index
    low_mask = bm["user_id"].isin(low_users)

    # Perform the 80-20 per-user train-val split, mixing in the
    # users that did not have their interactions split into the
    # training set
    splittable = bm[~low_mask]
    bm_train, bm_val = train_test_split(
        splittable,
        train_size=0.8,
        shuffle=True,
        stratify=splittable["user_id"],
        random_state=42
    )
    bm_train = pd.concat([bm_train, bm[low_mask]])

    return bm_train, bm_val # dataframes

def preprocess_item_categories():
    """
    Preprocesses item categories into a multi-hot-encoded tensor, s.t.
    across the 31 categories (cat 0 to cat 30) each item has a binary
    variable indicating whether it is of that category or not.

    Returns
        multi_hot: the item category matrix tensor

    multi_hot is of shape [10728, 31]:
    - 10728 items in KuaiRec
    - 31 categories as expected

    e.g. if item i is of categories 1, 2, 4 then row i is
    [1, 1, 0, 1, 0, 0, ...]

    Note that as these are already multi-hot, they won't need nn.Embedding
    layers to create vectors for them - they can go directly into nn.Linear.
    """

    # Read in the item categories; convert them to an actual Python
    # list using ast.literal_eval as they're encoded weirdly in the csv
    ic = pd.read_csv(DATA_DIR + "item_categories.csv")
    ic = ic["feat"].apply(literal_eval)

    # Set multi-hot position for each category, for each item (video)
    multi_hot = torch.zeros(10728, 31, dtype=torch.float32)
    for video_id, categories in ic.items():
        multi_hot[video_id, categories] = 1.0

    return multi_hot

# User features within KuaiRec's user_features.csv to handle
USER_FEATURE_COLS = {
    # Binary variables are of course 0 or 1, no missing data (no NaNs)
    "BINARY": [
        'is_lowactive_period',
        'is_live_streamer',
        'is_video_author'
    ],

    # One-hot features aren't actually one-hot vectors (misleading from KuaiRec)
    # but they are 0-indexed contiguous categorical variables, so they're suitable
    # for an nn.Embedding layer each. Some of them have missing data though so
    # they require NaN handling
    "ONEHOT": [f"onehot_feat{i}" for i in range(18)],

    # These buckets are categorical variables, but given as strings in the data
    # set so they need converting to contiguous 0-start indices; they do not have
    # any missing values so do not need any NaN handling
    "BUCKETED": [
        'user_active_degree',
        'follow_user_num_range',
        'fans_user_num_range',
        'friend_user_num_range',
        'register_days_range'
    ],

    # These are numeric values: they have long tails (as expected for data like follow
    # counts etc.) verified through data visualisation & so will use log1p scaling;
    # additionally will use z-score normalisation to debias model from variable means &
    # spread.
    "NUMERIC": [
        'follow_user_num',
        'fans_user_num',
        'friend_user_num',
        'register_days'
    ]
}

def preprocess_user_features():
    """
    Preprocesses user features, of which there are both categorical features and numeric
    features; they are handled as described in USER_FEATURE_COLS.

    These are returned as NumPy matrices rather than as tensors as they will need to be
    split up for batching anyway (and they will be converted to tensors then).

    Missing values for certain features (NaNs) are handled by adding an extra category
    where applicable for that feature.

    Returns
        (cat_feats, cat_sizes)
            cat_feats: 
                a NumPy array of shape [7176, 26]
                i.e. 26 categorical features for all 7176 users
            cat_sizes:
                the number of categories for each of the 26 features as a list

        numeric_feats:
            a NumPy array of shape [7176, 4]
            i.e. 4 normalised numeric features for all 7176 users

    """

    # Read in the user features
    uf = pd.read_csv(DATA_DIR + "user_features.csv")

    # Each feature is handled separately, but later need to become columns
    # s.t. each user is a row in the matrix containing their features.
    # Sizes are stored alongside the categoricals for nn.Embedding to know category sizes
    cat_cols, cat_sizes = [], []
    numeric_cols = []

    # Handle the binary features (BINARY)
    for col in USER_FEATURE_COLS["BINARY"]:
        cat_cols.append(uf[col].to_numpy())
        cat_sizes.append(2)
    
    # Handle the "one-hot" features (ONEHOT) that already are contiguous 0-indexed
    # categoricals, but that are seen to have missing values (NaNs) for some
    for col in USER_FEATURE_COLS["ONEHOT"]:
        # Get the max value: the number of categories = max + 1
        max_val = int(uf[col].dropna().max())

        # Does it have NaNs?
        has_nan = uf[col].isna().any()

        # If it has NaNs, an extra category is needed to denote this
        num_cats = (max_val + 1) + (1 if has_nan else 0)

        # Correct for NaNs: change NaNs to this new extra category if applicable
        vals = uf[col].fillna(max_val + 1).astype(int).to_numpy()

        # Add these corrected values to the list of columns
        cat_cols.append(vals)
        cat_sizes.append(num_cats)

    # Handle the bucketed features (BUCKETED): their strings need converting
    # to contiguous indices starting at 0
    for col in USER_FEATURE_COLS["BUCKETED"]:
        cat = pd.Categorical(uf[col])
        cat_cols.append(cat.codes)
        cat_sizes.append(len(cat.categories))

    # Handle the numeric values (NUMERIC): these need normalising
    for col in USER_FEATURE_COLS["NUMERIC"]:
        # Use log1p scaling to account for long tail, make more symmetric to reduce bias
        vals = np.log1p(uf[col].to_numpy(dtype=np.float32))

        # z-score normalisation: centre mean to 0, make std deviation 1
        # This reduces model bias towards variables with different means & spreads
        vals = (vals - vals.mean()) / (vals.std())

        numeric_cols.append(vals)
    
    # Stack the features into 2 matrices (categorical and numeric)
    cat_feats = np.stack(cat_cols, axis=1)
    numeric_feats = np.stack(numeric_cols, axis=1)

    return (cat_feats, cat_sizes), numeric_feats

class KRBig(Dataset):
    """
    PyTorch Dataset class for KuaiRec-Big; this allows PyTorch to create shuffled batches
    to train the model on; each retrieved row (by index idx) has:
        - user_id: user ID of this positive interaction
        - item_id: item ID of this positive interaction
        - user_cat_feats: the categorical features for this user
        - user_numeric_feats: the numeric features for this user
        - item_categories: the multi-hot vector denoting the item's categories

    Constructor parameters
        positive_interactions: the positive user-item interactions DataFrame
        user_cat_feats: a matrix of all categorical user features
        user_numeric_feats: a matrix of all numeric user features
        item_categories: a matrix of all multi-hot-encoded item categories
    """
    def __init__(
        self, 
        positive_interactions, 
        user_cat_feats, 
        user_numeric_feats,
        item_categories
    ):
        self.user_ids = positive_interactions["user_id"].to_numpy(dtype=np.int64)
        self.item_ids = positive_interactions["video_id"].to_numpy(dtype=np.int64)

        self.user_cat_feats = user_cat_feats
        self.user_numeric_feats = user_numeric_feats
        self.item_categories = item_categories

        # Useful to precompute for knowing exactly which users appear in the training & validation sets
        self.unique_user_ids = np.unique(self.user_ids)

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]

        return {
            "user_id": user_id,
            "item_id": item_id,
            "user_cat_feats": self.user_cat_feats[user_id],
            "user_numeric_feats": self.user_numeric_feats[user_id],
            "item_categories": self.item_categories[item_id]
        }