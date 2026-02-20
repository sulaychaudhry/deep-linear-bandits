import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def load_kuairec_big():
    bm = pd.read_csv(
        '/home/sulay/deep-linear-bandits/kuairec/data/big_matrix.csv',
        usecols=[
            'user_id',
            'video_id',
            'watch_ratio'
        ]
    ).sort_values(by=['user_id', 'video_id'])

    #print(bm)

    #print(bm[bm.watch_ratio >= 2.0])

    bm = bm[bm.watch_ratio >= 2.0].drop(columns=['watch_ratio']).drop_duplicates()

    #print(bm)

    #print(bm.groupby(by='user_id')['user_id'].count().sort_values())
    #print(bm.groupby(by='user_id')['user_id'].count().mean())
    #print(bm.groupby(by='user_id')['user_id'].count().median())

    #print(bm['user_id'].unique())
    #print(len(bm['user_id'].unique()))

    #print(np.sort(bm['video_id'].unique()))
    #print(len(bm['video_id'].unique()))

    return KRDataset(bm)

class KRDataset(Dataset):
    def __init__(self, data):
        self.user_ids = data['user_id'].to_numpy()
        self.item_ids = data['video_id'].to_numpy()

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx]
        }