import torch
from deep_linear_bandits.data import load_kuairec_big

def main():
    bm_train, bm_val = load_kuairec_big('/home/sulay/deep-linear-bandits/kuairec/data/big_matrix.csv')

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"