"""
Handles loading & preprocessing the KuaiRec dataset.
"""



from torch.utils.data import Dataset

class KRBig(Dataset):
    def __init__(self, pandas_table):
        self.d